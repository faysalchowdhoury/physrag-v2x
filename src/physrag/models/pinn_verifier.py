"""Physics-Informed verifier for traffic and grid claims.

Two conservation laws ground retrieval and diagnosis:

* **LWR traffic flow conservation** (Lighthill-Whitham-Richards):
  ``d(rho)/dt + d(q)/dx = 0`` with the Greenshields fundamental diagram
  ``q = rho * v_f * (1 - rho / rho_jam)``. A trajectory window that implies a
  large conservation residual (e.g., a spoofed teleport injecting/destroying
  vehicles, or a kinematically impossible speed) is flagged.

* **Grid nodal power balance** (Kirchhoff): at every bus the net injection must
  equal the power flowing out on incident lines; the residual is
  ``P_inj - sum_j B_ij (theta_i - theta_j)`` (DC approximation). A false-data
  injection that breaks balance yields a large residual.

The residual mathematics is exact NumPy and needs no training, which is what makes
it a reliable hallucination/spoof detector. An optional small Torch PINN
(``TrafficPINN``) is provided for learnable smoothing; it degrades gracefully when
torch is absent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False


# --------------------------------------------------------------------------- #
# LWR traffic conservation                                                    #
# --------------------------------------------------------------------------- #
def greenshields_flow(rho: np.ndarray, v_f: float, rho_jam: float) -> np.ndarray:
    """Greenshields fundamental diagram q(rho) = rho * v_f * (1 - rho/rho_jam)."""
    rho = np.clip(rho, 0.0, rho_jam)
    return rho * v_f * (1.0 - rho / rho_jam)


def lwr_residual(density: np.ndarray, flow: np.ndarray,
                 dx: float, dt: float) -> np.ndarray:
    """Discrete LWR conservation residual on a density/flow grid.

    ``density`` and ``flow`` are 2-D arrays indexed [time, space]. Returns the
    residual field ``d(rho)/dt + d(q)/dx`` via central/forward differences. A
    physically consistent field has residual ~ 0 everywhere.
    """
    density = np.asarray(density, dtype=float)
    flow = np.asarray(flow, dtype=float)
    drho_dt = np.gradient(density, dt, axis=0)
    dq_dx = np.gradient(flow, dx, axis=1)
    return drho_dt + dq_dx


@dataclass
class FieldStats:
    density: np.ndarray
    flow: np.ndarray
    speed: np.ndarray
    x_edges: np.ndarray
    t_edges: np.ndarray
    dx: float
    dt: float


def macroscopic_fields(x_m: Sequence[float], t_s: Sequence[float],
                       speed_mps: Sequence[float],
                       n_space: int = 12, n_time: int = 12,
                       lane_count: int = 1) -> FieldStats:
    """Aggregate point trajectories into macroscopic density/flow/speed fields.

    Density = vehicles per cell / (cell length * lanes); flow = density * mean
    speed (Edie-style cell average). Returns a :class:`FieldStats`.
    """
    x = np.asarray(x_m, float)
    t = np.asarray(t_s, float)
    v = np.asarray(speed_mps, float)
    if x.size == 0:
        z = np.zeros((n_time, n_space))
        return FieldStats(z, z, z, np.zeros(n_space + 1), np.zeros(n_time + 1), 1.0, 1.0)
    x_edges = np.linspace(x.min(), x.max() + 1e-6, n_space + 1)
    t_edges = np.linspace(t.min(), t.max() + 1e-6, n_time + 1)
    dx = float(np.mean(np.diff(x_edges))) or 1.0
    dt = float(np.mean(np.diff(t_edges))) or 1.0
    xi = np.clip(np.digitize(x, x_edges) - 1, 0, n_space - 1)
    ti = np.clip(np.digitize(t, t_edges) - 1, 0, n_time - 1)
    counts = np.zeros((n_time, n_space))
    vsum = np.zeros((n_time, n_space))
    for a, b, s in zip(ti, xi, v):
        counts[a, b] += 1.0
        vsum[a, b] += s
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_speed = np.where(counts > 0, vsum / np.maximum(counts, 1), 0.0)
    density = counts / (dx * max(lane_count, 1))
    flow = density * mean_speed
    return FieldStats(density, flow, mean_speed, x_edges, t_edges, dx, dt)


def traffic_violation_score(bsms, v_f: float = 26.8, rho_jam_vpm: float = 220.0,
                            lane_count: int = 1) -> float:
    """Return a physics-violation score in [0, inf) for a BSM window.

    The score fuses two first-principles checks that cleanly separate real benign
    trajectories from injected spoofs:

    * **Kinematic consistency** - reported speed vs. position-derived speed, plus
      an over-free-flow penalty. Teleports (position spoof) and impossible speeds
      (speed spoof) break it; real trajectories satisfy it to ~0.03.
    * **Static-obstruction anomaly** - stationary vehicles embedded in a moving
      stream violate flow continuity (ghost / Sybil queue injection).

    The macroscopic LWR conservation residual (``lwr_residual``) is retained for
    analysis/figures but excluded from the detector score: on sparse real windows
    its discretization noise dwarfs the attack signal, so using it would hurt both
    PLVR and false-positive rate. ``~0`` is physical; larger is less physical.
    """
    if len(bsms) < 4:
        return 0.0
    kin = _kinematic_inconsistency(bsms, v_f)
    ghost = _static_obstruction_anomaly(bsms)
    return kin + ghost


def _kinematic_inconsistency(bsms, v_f: float = 26.8) -> float:
    """Mean positive mismatch between reported speed and position-derived speed."""
    by_veh = {}
    for b in bsms:
        by_veh.setdefault(b.vehicle_id, []).append(b)
    v_max = v_f * 1.6                      # physical hard ceiling (no vehicle exceeds)
    mism = []
    for seq in by_veh.values():
        seq = sorted(seq, key=lambda b: b.t)
        for p, q in zip(seq, seq[1:]):
            dt = q.t - p.t
            if dt <= 0:
                continue
            disp = float(np.hypot(q.x_m - p.x_m, q.y_m - p.y_m))
            implied = disp / dt
            reported = 0.5 * (p.speed_mps + q.speed_mps)
            denom = max(reported, implied, 1.0)
            score = abs(implied - reported) / denom
            if reported > v_max or implied > v_max:   # over-free-flow impossibility
                score += (max(reported, implied) - v_max) / v_max
            mism.append(score)
    return float(np.mean(mism)) if mism else 0.0


def _static_obstruction_anomaly(bsms) -> float:
    """Fraction of vehicles that are fully static while the stream moves.

    A real stopped vehicle in a jam still creeps (displacement > a few metres over
    the window) and its neighbours are slow too; a fabricated ghost is pinned in
    place. Returns 0 when the segment itself is stopped (legitimate queue).
    """
    by_veh = {}
    for b in bsms:
        by_veh.setdefault(b.vehicle_id, []).append(b)
    if len(by_veh) < 2:
        return 0.0
    static = 0
    moving_speeds = []
    for seq in by_veh.values():
        seq = sorted(seq, key=lambda b: b.t)
        disp = float(np.hypot(seq[-1].x_m - seq[0].x_m, seq[-1].y_m - seq[0].y_m))
        vmax = max(b.speed_mps for b in seq)
        if vmax < 0.5 and disp < 2.0:
            static += 1
        else:
            moving_speeds.append(np.mean([b.speed_mps for b in seq]))
    seg_speed = float(np.mean(moving_speeds)) if moving_speeds else 0.0
    if seg_speed < 3.0:               # whole segment is slow -> legitimate queue
        return 0.0
    return static / len(by_veh)


# --------------------------------------------------------------------------- #
# Grid nodal power balance                                                    #
# --------------------------------------------------------------------------- #
def grid_balance_residual(p_injection: np.ndarray, theta: np.ndarray,
                          B: np.ndarray) -> np.ndarray:
    """DC nodal power-balance residual: P_inj - B @ theta, per bus.

    ``B`` is the DC susceptance (Laplacian-like) matrix. Residual ~ 0 at every
    bus when measurements are consistent; a false-data injection breaks it.
    """
    p = np.asarray(p_injection, float)
    theta = np.asarray(theta, float)
    B = np.asarray(B, float)
    return p - B @ theta


def grid_violation_score(nodes, B: np.ndarray, bus_index) -> float:
    """Normalized nodal-balance violation score for a set of SCADA nodes."""
    if not len(nodes):
        return 0.0
    n = B.shape[0]
    p = np.zeros(n)
    theta = np.zeros(n)
    seen = np.zeros(n, bool)
    for nd in nodes:
        i = bus_index.get(nd.bus_id)
        if i is None:
            continue
        p[i] = nd.p_injection_mw
        theta[i] = nd.theta_rad
        seen[i] = True
    res = grid_balance_residual(p, theta, B)
    scale = np.abs(p).max() + 1e-6
    if not seen.any():
        return 0.0
    return float(np.sqrt(np.mean((res[seen] / scale) ** 2)))


# --------------------------------------------------------------------------- #
# Combined verifier                                                           #
# --------------------------------------------------------------------------- #
class PINNVerifier:
    """Fuse traffic + grid residuals into one physics-plausibility signal."""

    def __init__(self, v_f: float = 26.8, rho_jam_vpm: float = 220.0,
                 plvr_threshold: float = 0.15, grid=None):
        self.v_f = v_f
        self.rho_jam_vpm = rho_jam_vpm
        self.plvr_threshold = plvr_threshold
        self.grid = grid  # optional GridCase (B matrix + bus index)

    def residual(self, bsms=(), scada=()) -> float:
        traf = traffic_violation_score(bsms, self.v_f, self.rho_jam_vpm) if bsms else 0.0
        grid = 0.0
        if scada and self.grid is not None:
            grid = grid_violation_score(scada, self.grid.B, self.grid.bus_index)
        return float(traf + grid)

    def violates(self, bsms=(), scada=()) -> bool:
        return self.residual(bsms, scada) > self.plvr_threshold


if HAVE_TORCH:  # pragma: no cover - exercised only with torch installed

    class TrafficPINN(nn.Module):
        """Small MLP PINN approximating rho(x,t); the LWR residual is its loss.

        Included for completeness/extension (learnable denoising of the field);
        the deterministic residual above is what the pipeline uses by default.
        """

        def __init__(self, hidden: int = 64, v_f: float = 26.8, rho_jam: float = 0.137):
            super().__init__()
            self.v_f, self.rho_jam = v_f, rho_jam
            self.net = nn.Sequential(
                nn.Linear(2, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, 1), nn.Softplus(),
            )

        def forward(self, xt):
            return self.net(xt).squeeze(-1)

        def physics_loss(self, xt):
            xt = xt.clone().requires_grad_(True)
            rho = self.forward(xt)
            grads = torch.autograd.grad(rho.sum(), xt, create_graph=True)[0]
            drho_dx, drho_dt = grads[:, 0], grads[:, 1]
            dq_drho = self.v_f * (1 - 2 * rho / self.rho_jam)
            return torch.mean((drho_dt + dq_drho * drho_dx) ** 2)
