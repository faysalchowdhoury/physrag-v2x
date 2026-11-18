"""Publication-grade figures (300 DPI PNG + vector PDF/SVG) from the ablation run.

Generates:
  fig_ablation_metrics.{png,pdf,svg}  - PLVR / Diagnosis F1 / Faithfulness by method
  fig_tail_latency.{png,pdf,svg}      - P50/P95/P99 tail latency by method
  fig_physics_separation.{png,pdf,svg}- benign vs attack physics-residual separation

Run ``experiments/run_ablations.py`` first (writes results/ablation_results.json).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIGDIR = ROOT / "results" / "figures"
MODES = ["base_vlm", "standard_rag", "graph_rag", "phys_rag"]
COLORS = {"base_vlm": "#9aa5b1", "standard_rag": "#4a6fa5",
          "graph_rag": "#2e8b57", "phys_rag": "#c0392b"}


def _style():
    for fam in ("Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"):
        matplotlib.rcParams["font.serif"] = [fam]
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.size"] = 11
    matplotlib.rcParams["axes.titleweight"] = "bold"


def _save(fig, stem):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGDIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_metrics(res):
    labels = [res["modes"][m]["label"] for m in MODES]
    metrics = [("PLVR", "PLVR (lower better)"), ("diagnosis_f1", "Diagnosis F1"),
               ("faithfulness", "Faithfulness")]
    x = np.arange(len(MODES))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for j, (key, lbl) in enumerate(metrics):
        vals = [res["modes"][m][key] for m in MODES]
        ax.bar(x + (j - 1) * w, vals, w, label=lbl,
               color=["#c0392b", "#4a6fa5", "#2e8b57"][j], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Ablation: physics grounding drives PLVR, F1 and faithfulness")
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    _save(fig, "fig_ablation_metrics")


def fig_latency(res):
    labels = [res["modes"][m]["label"] for m in MODES]
    x = np.arange(len(MODES))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for j, key in enumerate(("p50_ms", "p95_ms", "p99_ms")):
        vals = [res["modes"][m][key] for m in MODES]
        ax.bar(x + (j - 1) * w, vals, w, label=key.upper().replace("_MS", ""),
               color=["#8fb0d8", "#4a6fa5", "#1f3864"][j], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Latency per diagnosis (ms)")
    ax.set_title("Tail latency: physics verification adds a bounded, principled cost")
    ax.legend(frameon=False, fontsize=9, title="percentile")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    _save(fig, "fig_tail_latency")


def fig_physics_separation():
    from physrag.config import Settings
    from physrag.etl.synthetic_generator import build_incidents
    from physrag.pipeline import PhysRAGPipeline
    cfg = Settings(plvr_threshold=0.08)
    inc = build_incidents(n_windows=16, attack_fraction=0.5, seed=0)
    pipe = PhysRAGPipeline(settings=cfg)
    pipe.ingest_many(inc)
    ben = [pipe.oracle_residual(i.incident_id) for i in inc if not i.is_attack]
    atk = [pipe.oracle_residual(i.incident_id) for i in inc if i.is_attack]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter([0] * len(ben), ben, s=60, color="#2e8b57", label="benign (real NGSIM)",
               zorder=3)
    ax.scatter([1] * len(atk), atk, s=60, color="#c0392b",
               label="injected attack", marker="^", zorder=3)
    ax.axhline(cfg.plvr_threshold, ls="--", color="#555",
               label=f"PLVR threshold = {cfg.plvr_threshold}")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["benign", "attack"])
    ax.set_ylabel("PINN physics residual")
    ax.set_title("Physics residual separates real benign traffic from injected attacks")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    _save(fig, "fig_physics_separation")


def main():
    _style()
    res_path = ROOT / "results" / "ablation_results.json"
    if not res_path.exists():
        raise SystemExit("run experiments/run_ablations.py first")
    res = json.loads(res_path.read_text())
    fig_metrics(res)
    fig_latency(res)
    fig_physics_separation()
    print("wrote figures to", FIGDIR)
    for p in sorted(FIGDIR.glob("*.png")):
        print("  ", p.name, p.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
