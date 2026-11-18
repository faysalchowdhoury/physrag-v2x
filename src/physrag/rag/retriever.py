"""Two-stage physics-aware hybrid retriever (the PhysRAG core).

Pipeline:
  1. **Dense recall** — HNSW/vector search for top-k candidate nodes.
  2. **Graph expansion** — bounded multi-source Dijkstra (C++ core or Python
     fallback) grows a causal subgraph around the candidates.
  3. **PINN re-rank** — each candidate's score is penalized by its physics
     residual (LWR traffic / grid balance), demoting physically implausible
     (spoofed/hallucinated) evidence.
  4. **Knapsack packing** — a 0-1 knapsack packs the highest physics-penalized
     value into the token budget.

The ablation variants (`standard_rag`, `graph_rag`, `phys_rag`) switch stages on
and off so experiments isolate each contribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..algorithms import knapsack_pack, multi_source_dijkstra_subgraph
from ..config import Settings, get_settings
from ..models.pinn_verifier import PINNVerifier
from ..models.schemas import Incident, RetrievedItem
from .graph_index import GraphIndex


def estimate_tokens(text: str) -> int:
    """~4 chars/token heuristic (matches common BPE density)."""
    return max(1, len(text) // 4)


@dataclass
class RetrievalResult:
    items: List[RetrievedItem]
    packed_value: float
    subgraph_size: int
    mode: str


class HybridRetriever:
    def __init__(self, index: GraphIndex, verifier: Optional[PINNVerifier] = None,
                 settings: Optional[Settings] = None,
                 incidents: Optional[Dict[str, Incident]] = None):
        self.index = index
        self.cfg = settings or get_settings()
        self.verifier = verifier or PINNVerifier(plvr_threshold=self.cfg.plvr_threshold)
        # NB: `incidents or {}` would replace a (falsy) empty dict with a new one,
        # severing the shared reference the pipeline mutates on ingest.
        self.incidents = incidents if incidents is not None else {}
        self._resid_cache: Dict[str, float] = {}

    # -- stages ---------------------------------------------------------------
    def _recall(self, query: str, top_k: int):
        return self.index.vector_search(query, top_k)

    def _expand(self, seeds: Sequence[str], hops: int):
        reached = multi_source_dijkstra_subgraph(self.index.graph, list(seeds), hops)
        return reached  # {node: (dist, hops)}

    def _residual_for(self, node_id: str) -> float:
        inc = self.incidents.get(node_id)
        if inc is None:
            return 0.0
        cached = self._resid_cache.get(node_id)
        if cached is None:
            cached = self.verifier.residual(inc.bsms, inc.scada)
            self._resid_cache[node_id] = cached
        return cached

    # -- public modes ---------------------------------------------------------
    def retrieve(self, query: str, mode: str = "phys_rag") -> RetrievalResult:
        """Run retrieval in one of: 'standard_rag', 'graph_rag', 'phys_rag'."""
        recalled = self._recall(query, self.cfg.hnsw_top_k)
        cand: Dict[str, float] = {nid: sc for nid, sc in recalled}
        subgraph_size = len(cand)

        if mode in ("graph_rag", "phys_rag"):
            reached = self._expand(list(cand.keys()), self.cfg.graph_hops)
            subgraph_size = len(reached)
            for nid, (dist, hops) in reached.items():
                base = cand.get(nid, 0.3)               # expanded nodes get a prior
                cand[nid] = max(cand.get(nid, 0.0), base * (1.0 / (1.0 + dist)))

        tokens, relevance, residuals, ids = [], [], [], []
        for nid, rel in cand.items():
            text = self.index.text(nid)
            resid = self._residual_for(nid) if mode == "phys_rag" else 0.0
            tokens.append(estimate_tokens(text))
            relevance.append(rel)
            residuals.append(resid)
            ids.append(nid)

        lam = self.cfg.physics_lambda if mode == "phys_rag" else 0.0
        chosen, value = knapsack_pack(tokens, relevance, residuals,
                                      self.cfg.token_budget, lam)

        items = [RetrievedItem(
            item_id=ids[i], text=self.index.text(ids[i]), tokens=tokens[i],
            relevance=relevance[i], physics_residual=residuals[i],
            hops=0) for i in chosen]
        items.sort(key=lambda it: (it.relevance - lam * it.physics_residual),
                   reverse=True)
        return RetrievalResult(items=items, packed_value=value,
                               subgraph_size=subgraph_size, mode=mode)
