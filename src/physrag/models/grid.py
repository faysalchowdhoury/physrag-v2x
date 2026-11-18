"""Real IEEE 14-bus benchmark power system (DC model).

The IEEE 14-bus test case is canonical, publicly documented power-system data
(14 buses, 20 branches; standard branch reactances below). We build its DC
susceptance matrix ``B`` so the PINN verifier can check nodal power balance
``P_inj = B @ theta`` against SCADA measurements and flag false-data injection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# Standard IEEE 14-bus branch reactances (from_bus, to_bus, x_pu).
IEEE14_BRANCHES: List[Tuple[int, int, float]] = [
    (1, 2, 0.05917), (1, 5, 0.22304), (2, 3, 0.19797), (2, 4, 0.17632),
    (2, 5, 0.17388), (3, 4, 0.17103), (4, 5, 0.04211), (4, 7, 0.20912),
    (4, 9, 0.55618), (5, 6, 0.25202), (6, 11, 0.19890), (6, 12, 0.25581),
    (6, 13, 0.13027), (7, 8, 0.17615), (7, 9, 0.11001), (9, 10, 0.08450),
    (9, 14, 0.27038), (10, 11, 0.19207), (12, 13, 0.19988), (13, 14, 0.34802),
]
# Nominal real-power loads per bus (MW), IEEE 14-bus base case.
IEEE14_LOAD_MW: Dict[int, float] = {
    1: 0.0, 2: 21.7, 3: 94.2, 4: 47.8, 5: 7.6, 6: 11.2, 7: 0.0,
    8: 0.0, 9: 29.5, 10: 9.0, 11: 3.5, 12: 6.1, 13: 13.5, 14: 14.9,
}
# Buses with generation (slack + PV) in the base case.
IEEE14_GEN_BUSES = (1, 2, 3, 6, 8)


@dataclass
class GridCase:
    n_bus: int
    branches: List[Tuple[int, int, float]]
    bus_ids: List[int]
    bus_index: Dict[int, int]
    B: np.ndarray = field(repr=False)
    load_mw: Dict[int, float] = field(default_factory=dict)

    def incident_buses(self, bus_id: int) -> List[int]:
        return [b for (a, b, _) in self.branches if a == bus_id] + \
               [a for (a, b, _) in self.branches if b == bus_id]


def build_ieee14() -> GridCase:
    """Construct the IEEE 14-bus DC case with its susceptance matrix."""
    bus_ids = list(range(1, 15))
    idx = {b: i for i, b in enumerate(bus_ids)}
    n = len(bus_ids)
    B = np.zeros((n, n))
    for a, b, x in IEEE14_BRANCHES:
        bij = 1.0 / x
        ia, ib = idx[a], idx[b]
        B[ia, ia] += bij
        B[ib, ib] += bij
        B[ia, ib] -= bij
        B[ib, ia] -= bij
    return GridCase(n_bus=n, branches=list(IEEE14_BRANCHES), bus_ids=bus_ids,
                    bus_index=idx, B=B, load_mw=dict(IEEE14_LOAD_MW))


def consistent_injection(grid: GridCase, theta: np.ndarray) -> np.ndarray:
    """Return the nodal injection exactly consistent with angles (P = B @ theta)."""
    return grid.B @ np.asarray(theta, float)
