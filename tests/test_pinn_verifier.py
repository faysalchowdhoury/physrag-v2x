"""Tests for the physics verifier on real vs. spoofed/hallucinated data."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physrag.models.grid import build_ieee14, consistent_injection
from physrag.models.pinn_verifier import (PINNVerifier, grid_balance_residual,
                                          greenshields_flow, lwr_residual,
                                          traffic_violation_score)
from physrag.models.schemas import AttackLabel, BSM, SCADANode
from physrag.etl.synthetic_generator import (build_incidents, inject_bsm_attacks,
                                             load_ngsim)


def _benign_track(vehicle="v1", v=20.0, n=10, dt=0.1):
    """A kinematically consistent track: position advances at the reported speed."""
    return [BSM(vehicle_id=vehicle, t=i * dt, x_m=0.0, y_m=v * i * dt,
                speed_mps=v) for i in range(n)]


# --- traffic conservation ----------------------------------------------------

def test_benign_track_is_physical():
    score = traffic_violation_score(_benign_track())
    assert score < 0.05


def test_position_spoof_flagged():
    track = _benign_track()
    track[5] = track[5].model_copy(update={"y_m": track[5].y_m + 80.0,
                                           "label": AttackLabel.POSITION_SPOOF})
    assert traffic_violation_score(track) > traffic_violation_score(_benign_track())
    assert traffic_violation_score(track) > 0.1


def test_speed_spoof_flagged():
    track = _benign_track()
    track[5] = track[5].model_copy(update={"speed_mps": 120.0,
                                           "label": AttackLabel.SPEED_SPOOF})
    assert traffic_violation_score(track) > 0.1


def test_greenshields_and_lwr_shapes():
    rho = np.linspace(0, 220, 12)            # 0 .. rho_jam
    q = greenshields_flow(rho, v_f=26.8, rho_jam=220.0)
    assert q[0] == 0 and q[-1] == pytest.approx(0, abs=1e-6)   # 0 at empty and jam
    assert q.max() > 0 and np.argmax(q) == len(q) // 2         # peak at rho_jam/2
    dens = np.ones((5, 5)); flow = np.ones((5, 5))
    res = lwr_residual(dens, flow, dx=1.0, dt=1.0)
    assert res.shape == (5, 5)
    assert np.allclose(res, 0.0)             # constant field conserves exactly


# --- grid balance ------------------------------------------------------------

def test_grid_consistent_injection_has_zero_residual():
    grid = build_ieee14()
    theta = np.random.default_rng(0).normal(0, 0.1, grid.n_bus)
    p = consistent_injection(grid, theta)
    res = grid_balance_residual(p, theta, grid.B)
    assert np.allclose(res, 0.0, atol=1e-9)


def test_grid_fdi_breaks_balance():
    grid = build_ieee14()
    theta = np.random.default_rng(1).normal(0, 0.1, grid.n_bus)
    p = consistent_injection(grid, theta)
    p[3] += 0.8                               # false-data injection at a bus
    res = grid_balance_residual(p, theta, grid.B)
    assert abs(res[3]) > 0.5


# --- combined verifier on real + injected data ------------------------------

def test_verifier_separates_real_benign_from_injected_attacks():
    real = load_ngsim(max_rows=4000)
    assert real, "real NGSIM fixture must load"
    benign_score = traffic_violation_score(real[:500])
    attacked = inject_bsm_attacks(real[:500], rate=0.3, seed=0)
    attack_score = traffic_violation_score(attacked)
    assert attack_score > benign_score
    assert benign_score < 0.08 <= attack_score


def test_pinn_verifier_flags_attack_incidents():
    incidents = build_incidents(n_windows=8, attack_fraction=0.5, seed=0)
    grid = build_ieee14()
    verifier = PINNVerifier(plvr_threshold=0.08, grid=grid)
    for inc in incidents:
        flagged = verifier.violates(inc.bsms, inc.scada)
        assert flagged == inc.is_attack, (inc.incident_id, inc.labels)
