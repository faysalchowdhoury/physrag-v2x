"""Ablation harness: Base VLM vs Standard RAG vs GraphRAG vs PhysRAG-V2X.

Metrics (computed from the real NGSIM + IEEE-14 pipeline, with injected labeled
attacks as ground truth):

  PLVR               Physical Law Violation Rate -- fraction of incidents where a
                     real physics violation is present but the system fails to flag
                     it (lower is better).
  Diagnosis F1       F1 of the attack-vs-benign decision against ground truth.
  Faithfulness       fraction of packed context items that are physically valid
                     (oracle residual <= threshold) -- i.e., not spoofed.
  CauseRecall        fraction of incidents whose packed context includes a causal
                     graph neighbor (retrieval of the true cause).
  P50/P95/P99        tail latency (ms) over repeated diagnoses.

Writes results/ablation_results.json and prints a table.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physrag.config import Settings
from physrag.etl.synthetic_generator import build_incidents
from physrag.models.schemas import AttackLabel
from physrag.pipeline import PhysRAGPipeline

MODES = ["base_vlm", "standard_rag", "graph_rag", "phys_rag"]
MODE_LABEL = {"base_vlm": "Base VLM", "standard_rag": "Standard RAG",
              "graph_rag": "GraphRAG", "phys_rag": "PhysRAG-V2X"}


def _f1(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def evaluate(n_windows: int = 16, repeats: int = 20, seed: int = 0) -> dict:
    # Experiment settings create selection pressure on the small committed fixture
    # (tight token budget + limited first-stage recall) so the graph-expansion and
    # physics-rerank contributions are measurable; a production deployment uses the
    # larger defaults in physrag.config.
    cfg = Settings(token_budget=220, hnsw_top_k=4, graph_hops=2,
                   physics_lambda=8.0, plvr_threshold=0.08)
    incidents = build_incidents(n_windows=n_windows, attack_fraction=0.5, seed=seed)
    pipe = PhysRAGPipeline(settings=cfg)
    pipe.ingest_many(incidents)
    thr = cfg.plvr_threshold

    truth = {i.incident_id: i.is_attack for i in incidents}
    oracle = {i.incident_id: pipe.oracle_residual(i.incident_id) for i in incidents}
    neighbors = {i.incident_id: {d for d, _ in pipe.index.graph.get(i.incident_id, [])}
                 for i in incidents}

    results = {}
    for mode in MODES:
        tp = fp = fn = tn = 0
        leaks = 0
        faith_num = faith_den = 0
        cause_hits = cause_total = 0
        latencies = []
        for inc in incidents:
            resp, context = pipe.diagnose_ex(inc.incident_id, mode=mode)
            pred = resp.violates_physics
            is_atk = truth[inc.incident_id]
            tp += int(pred and is_atk); fp += int(pred and not is_atk)
            fn += int(not pred and is_atk); tn += int(not pred and not is_atk)
            # PLVR: real violation present (oracle) but unflagged
            if oracle[inc.incident_id] > thr and not pred:
                leaks += 1
            # Faithfulness over packed context (oracle residual of cited incidents)
            for it in context:
                faith_den += 1
                faith_num += int(oracle.get(it.item_id, 0.0) <= thr)
            # Cause recall
            if context:
                cause_total += 1
                ids = {it.item_id for it in context}
                if ids & neighbors[inc.incident_id]:
                    cause_hits += 1
        # latency distribution (repeat diagnoses)
        for _ in range(repeats):
            for inc in incidents:
                t0 = time.perf_counter()
                pipe.diagnose_ex(inc.incident_id, mode=mode)
                latencies.append((time.perf_counter() - t0) * 1000)
        latencies.sort()

        def pct(p):
            k = min(len(latencies) - 1, int(round(p / 100 * (len(latencies) - 1))))
            return latencies[k]

        results[mode] = {
            "label": MODE_LABEL[mode],
            "PLVR": round(leaks / len(incidents), 4),
            "diagnosis_f1": round(_f1(tp, fp, fn), 4),
            "faithfulness": round(faith_num / faith_den, 4) if faith_den else 0.0,
            "cause_recall": round(cause_hits / cause_total, 4) if cause_total else 0.0,
            "p50_ms": round(pct(50), 4), "p95_ms": round(pct(95), 4),
            "p99_ms": round(pct(99), 4),
            "detection": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        }
    return {"n_incidents": len(incidents),
            "n_attacks": sum(truth.values()),
            "backend": __import__("physrag.algorithms", fromlist=["BACKEND"]).BACKEND,
            "modes": results}


def main():
    out = evaluate()
    outdir = ROOT / "results"
    outdir.mkdir(exist_ok=True)
    (outdir / "ablation_results.json").write_text(json.dumps(out, indent=2))

    print(f"\nReal NGSIM I-80 + IEEE 14-bus | {out['n_incidents']} incidents "
          f"({out['n_attacks']} attacks) | algo backend: {out['backend']}\n")
    hdr = f"{'Method':<15}{'PLVR':>8}{'Diag F1':>9}{'Faith':>8}{'CauseR':>8}" \
          f"{'P50':>8}{'P95':>8}{'P99':>8}"
    print(hdr); print("-" * len(hdr))
    for mode in MODES:
        r = out["modes"][mode]
        print(f"{r['label']:<15}{r['PLVR']:>8.3f}{r['diagnosis_f1']:>9.3f}"
              f"{r['faithfulness']:>8.3f}{r['cause_recall']:>8.3f}"
              f"{r['p50_ms']:>8.2f}{r['p95_ms']:>8.2f}{r['p99_ms']:>8.2f}")
    print(f"\nwrote {outdir / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
