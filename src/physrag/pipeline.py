"""End-to-end PhysRAG pipeline: ingest -> index -> retrieve -> verify -> diagnose.

This module is the orchestration shared by the FastAPI service and the tests, so
the full path (BSM/SCADA ingestion to a grounded, physics-verified diagnosis) is
exercised without needing the HTTP server, Redis, Spark or a GPU.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .algorithms import SpatialCache
from .config import Settings, get_settings
from .models.grid import GridCase, build_ieee14
from .models.pinn_verifier import PINNVerifier
from .models.schemas import BSM, DiagnosticResponse, Incident, SCADANode
from .rag.graph_index import GraphIndex
from .rag.retriever import HybridRetriever
from .rag.vlm_finetune import VLMDiagnoser


class TokenBucket:
    """Classic token-bucket rate limiter. allow() is O(1)."""

    def __init__(self, rate_per_s: float, burst: int, clock=time.monotonic):
        self.rate = rate_per_s
        self.capacity = burst
        self.tokens = float(burst)
        self._clock = clock
        self._t = clock()

    def allow(self, cost: float = 1.0) -> bool:
        now = self._clock()
        self.tokens = min(self.capacity, self.tokens + (now - self._t) * self.rate)
        self._t = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


def incident_summary(inc: Incident) -> str:
    ev = sum(n.ev_load_mw for n in inc.scada)
    return (f"{inc.scene_text} vehicles={len(inc.bsms)} grid_buses={len(inc.scada)} "
            f"ev_load={ev:.0f}MW severity={inc.severity:.2f} geokey={inc.geokey}")


@dataclass
class PhysRAGPipeline:
    settings: Settings = field(default_factory=get_settings)
    grid: GridCase = field(default_factory=build_ieee14)
    index: GraphIndex = field(default=None)   # type: ignore
    incidents: Dict[str, Incident] = field(default_factory=dict)

    def __post_init__(self):
        if self.index is None:
            self.index = GraphIndex(dim=self.settings.embed_dim,
                                    redis_url=None)  # memory by default
        self.verifier = PINNVerifier(plvr_threshold=self.settings.plvr_threshold,
                                     grid=self.grid)
        self.cache = SpatialCache(self.settings.cache_capacity, self.settings.cache_ttl_s)
        self.vlm = VLMDiagnoser()
        self.retriever = HybridRetriever(self.index, self.verifier, self.settings,
                                         self.incidents)

    # -- ingestion ------------------------------------------------------------
    def ingest_incident(self, inc: Incident) -> None:
        self.incidents[inc.incident_id] = inc
        self.index.add_node(inc.incident_id, incident_summary(inc),
                            meta={"geokey": inc.geokey, "t": inc.t_start})
        # spatial edges (same geokey) + temporal edges (time proximity)
        for other_id, other in self.incidents.items():
            if other_id == inc.incident_id:
                continue
            if other.geokey == inc.geokey:
                self.index.add_edge(inc.incident_id, other_id, weight=1.0)
            dt = abs(other.t_start - inc.t_start)
            if dt < 30.0:
                self.index.add_edge(inc.incident_id, other_id, weight=1.0 + dt)

    def ingest_many(self, incidents: Sequence[Incident]) -> None:
        for inc in incidents:
            self.ingest_incident(inc)

    # -- diagnosis ------------------------------------------------------------
    def diagnose(self, incident_id: str, mode: str = "phys_rag",
                 query: Optional[str] = None) -> DiagnosticResponse:
        return self.diagnose_ex(incident_id, mode, query)[0]

    def diagnose_ex(self, incident_id: str, mode: str = "phys_rag",
                    query: Optional[str] = None):
        """Return (response, retrieved_items). Only PhysRAG consults the physics
        verifier / residual re-rank; the other modes are physics-blind, which is
        what the ablation isolates."""
        t0 = time.perf_counter()
        inc = self.incidents[incident_id]
        q = query or inc.scene_text
        use_context = mode != "base_vlm"
        context = self.retriever.retrieve(q, mode=mode).items if use_context else []

        physics_aware = mode == "phys_rag"
        if physics_aware:
            sys_resid = self.verifier.residual(inc.bsms, inc.scada)
            sys_violates = sys_resid > self.settings.plvr_threshold
        else:
            sys_resid, sys_violates = 0.0, False
        resp = self.vlm.diagnose(inc, context, sys_resid, sys_violates,
                                 use_context=use_context)
        resp.latency_ms = (time.perf_counter() - t0) * 1000
        return resp, context

    def oracle_residual(self, incident_id: str) -> float:
        inc = self.incidents[incident_id]
        return self.verifier.residual(inc.bsms, inc.scada)
