"""Real-data loader + labeled perturbation generator.

Despite the historical name, this module is grounded in **real data**:

* Traffic / BSM streams come from the **real NGSIM I-80 trajectories** (committed
  fixture ``data/ngsim_i80_sample.csv.gz``; the full set is fetched from the USDOT
  ITS DataHub). Positions, speeds and accelerations are the measured values.
* The grid comes from the **real IEEE 14-bus benchmark** (``models.grid``); EV
  charging load is coupled to real traffic density.

On top of the real streams we inject **labeled** cyber-physical perturbations
(position/speed spoofing, ghost/Sybil vehicles, grid false-data injection). These
labels are the ground truth needed to measure the Physical Law Violation Rate —
the same "real base + injected labeled attack" methodology used throughout this
portfolio. CCTV ``scene_text`` is rendered from the real incident state, not
fabricated independently.
"""
from __future__ import annotations

import gzip
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..models.grid import GridCase, build_ieee14
from ..models.schemas import AttackLabel, BSM, Incident, SCADANode

FT_TO_M = 0.3048


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_ngsim(path: Optional[str] = None, max_rows: Optional[int] = None):
    """Load the real NGSIM trajectory fixture as a list of BSM objects (benign).

    Uses pandas if available; otherwise a stdlib CSV reader. Returns BSMs sorted
    by (vehicle, time) with SI units (metres, m/s).
    """
    p = Path(path) if path else _repo_root() / "data" / "ngsim_i80_sample.csv.gz"
    rows = _read_csv_gz(p, max_rows)
    bsms: List[BSM] = []
    for r in rows:
        try:
            t = float(r["global_time"]) / 1000.0           # ms -> s
            vx = float(r["v_vel"]) * FT_TO_M               # ft/s -> m/s
            bsms.append(BSM(
                vehicle_id=str(r["vehicle_id"]),
                t=t,
                x_m=float(r["local_x"]) * FT_TO_M,
                y_m=float(r["local_y"]) * FT_TO_M,
                speed_mps=max(0.0, vx),
                accel_mps2=float(r.get("v_acc", 0.0) or 0.0) * FT_TO_M,
                heading_deg=0.0,
                lane=int(float(r.get("lane_id", 0) or 0)),
                label=AttackLabel.BENIGN,
            ))
        except (KeyError, ValueError):
            continue
    bsms.sort(key=lambda b: (b.vehicle_id, b.t))
    return bsms


def _read_csv_gz(path: Path, max_rows: Optional[int]):
    try:
        import pandas as pd
        df = pd.read_csv(path)
        if max_rows:
            df = df.head(max_rows)
        return df.to_dict("records")
    except Exception:
        import csv
        op = gzip.open if str(path).endswith(".gz") else open
        with op(path, "rt", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            out = []
            for i, row in enumerate(reader):
                if max_rows and i >= max_rows:
                    break
                out.append(row)
            return out


def calibrate_kinematics(bsms: Sequence[BSM]) -> Dict[str, float]:
    """Measure real free-flow speed and speed spread from the NGSIM stream, so the
    injected attacks are scaled to the true data distribution."""
    sp = np.array([b.speed_mps for b in bsms], float)
    if sp.size == 0:
        return {"v_free_mps": 26.8, "v_p95_mps": 26.8, "v_std_mps": 5.0}
    return {
        "v_free_mps": float(np.percentile(sp[sp > 0], 85)) if (sp > 0).any() else 26.8,
        "v_p95_mps": float(np.percentile(sp, 95)),
        "v_std_mps": float(sp.std()) or 5.0,
    }


# --------------------------------------------------------------------------- #
# Labeled attack injection on the real BSM stream                             #
# --------------------------------------------------------------------------- #
def inject_bsm_attacks(bsms: Sequence[BSM], rate: float = 0.08,
                       seed: int = 0) -> List[BSM]:
    """Inject labeled position/speed spoofing and ghost vehicles into real BSMs.

    ``rate`` is the fraction of real messages perturbed; ghosts are added on top.
    Returns a new list (originals preserved, perturbed copies labeled).
    """
    rng = random.Random(seed)
    cal = calibrate_kinematics(bsms)
    out: List[BSM] = []
    for b in bsms:
        if rng.random() < rate:
            kind = rng.choice([AttackLabel.POSITION_SPOOF, AttackLabel.SPEED_SPOOF])
            if kind == AttackLabel.POSITION_SPOOF:
                jump = rng.uniform(40, 120)                # metres teleport
                out.append(b.model_copy(update={
                    "y_m": b.y_m + jump, "label": AttackLabel.POSITION_SPOOF}))
            else:
                out.append(b.model_copy(update={
                    "speed_mps": cal["v_p95_mps"] * rng.uniform(2.2, 3.5),
                    "label": AttackLabel.SPEED_SPOOF}))
        else:
            out.append(b)
    # Ghost (Sybil) vehicles: physically static clusters with fabricated ids.
    if bsms:
        t0 = min(b.t for b in bsms)
        xs = [b.x_m for b in bsms]
        ys = [b.y_m for b in bsms]
        for g in range(max(1, int(len(bsms) * rate * 0.3))):
            out.append(BSM(
                vehicle_id=f"ghost-{g}", t=t0 + rng.uniform(0, 5),
                x_m=rng.uniform(min(xs), max(xs)),
                y_m=rng.uniform(min(ys), max(ys)),
                speed_mps=0.0, accel_mps2=0.0, lane=rng.randint(1, 6),
                label=AttackLabel.GHOST))
    return out


# --------------------------------------------------------------------------- #
# Real grid + EV coupling + labeled false-data injection                      #
# --------------------------------------------------------------------------- #
def grid_snapshots(grid: GridCase, traffic_density: float, t: float,
                   fdi_bus: Optional[int] = None, seed: int = 0) -> List[SCADANode]:
    """Produce SCADA nodes consistent with the real IEEE 14-bus case, add an EV
    charging load proportional to real traffic density, and optionally inject a
    labeled false-data attack at ``fdi_bus``.
    """
    rng = np.random.default_rng(seed)
    n = grid.n_bus
    theta = rng.normal(0, 0.05, n)
    theta[0] = 0.0                                   # slack reference
    p = grid.B @ theta                               # consistent injections (pu-ish)
    nodes: List[SCADANode] = []
    ev = 2.0 + 8.0 * float(np.clip(traffic_density, 0, 1))   # MW, traffic-coupled
    for bid in grid.bus_ids:
        i = grid.bus_index[bid]
        label = AttackLabel.BENIGN
        p_meas = float(p[i])
        ev_here = ev if bid in (4, 5, 9) else 0.0    # EV-heavy buses near the road
        p_meas -= ev_here / 100.0
        if fdi_bus is not None and bid == fdi_bus:
            p_meas += 0.6                            # false data: breaks balance
            label = AttackLabel.GRID_FDI
        nodes.append(SCADANode(bus_id=bid, t=t, p_injection_mw=p_meas,
                               v_pu=1.0 + 0.02 * theta[i], theta_rad=float(theta[i]),
                               ev_load_mw=ev_here, label=label))
    return nodes


# --------------------------------------------------------------------------- #
# Scene text rendered from the real incident state                            #
# --------------------------------------------------------------------------- #
def render_scene_text(bsms: Sequence[BSM], scada: Sequence[SCADANode]) -> str:
    """Render a CCTV-style scene description from the real measured state."""
    if not bsms:
        return "No vehicles observed in the segment."
    speeds = np.array([b.speed_mps for b in bsms])
    stopped = int((speeds < 1.0).sum())
    mean_kmh = float(speeds.mean()) * 3.6
    ev = sum(n.ev_load_mw for n in scada)
    cond = ("heavy congestion" if mean_kmh < 15 else
            "moderate flow" if mean_kmh < 40 else "free flow")
    desc = (f"Camera shows {cond}: {len(bsms)} vehicles, {stopped} stopped, "
            f"mean speed {mean_kmh:.0f} km/h.")
    if ev > 0:
        desc += f" Nearby substations carry {ev:.0f} MW of EV charging load."
    return desc


def build_incidents(n_windows: int = 8, attack_fraction: float = 0.5,
                    seed: int = 0, ngsim_path: Optional[str] = None) -> List[Incident]:
    """Assemble fused multimodal incidents from real NGSIM + IEEE 14-bus, half of
    them carrying injected labeled attacks. This is the experiment corpus."""
    rng = random.Random(seed)
    base = load_ngsim(ngsim_path)
    grid = build_ieee14()
    if not base:
        return []
    t0, t1 = min(b.t for b in base), max(b.t for b in base)
    span = max(t1 - t0, 1e-6)
    incidents: List[Incident] = []
    for w in range(n_windows):
        a = t0 + span * w / n_windows
        b_ = t0 + span * (w + 1) / n_windows
        win = [x for x in base if a <= x.t < b_]
        if len(win) < 4:
            continue
        is_attack = rng.random() < attack_fraction
        fdi_bus = rng.choice((4, 5, 9)) if is_attack and rng.random() < 0.5 else None
        bsms = inject_bsm_attacks(win, rate=0.1, seed=seed + w) if is_attack else list(win)
        dens = min(1.0, len(win) / 600.0)
        scada = grid_snapshots(grid, dens, a, fdi_bus=fdi_bus, seed=seed + w)
        labels = sorted({x.label for x in bsms if x.label != AttackLabel.BENIGN} |
                        {s.label for s in scada if s.label != AttackLabel.BENIGN})
        inc = Incident(
            incident_id=f"inc-{w:03d}", t_start=a, t_end=b_,
            geokey=bsms[0].geokey() if bsms else "0:0",
            bsms=bsms, scada=scada,
            scene_text=render_scene_text(bsms, scada),
            severity=dens, labels=list(labels) or [AttackLabel.BENIGN],
        )
        incidents.append(inc)
    return incidents
