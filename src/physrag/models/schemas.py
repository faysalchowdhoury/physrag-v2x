"""Typed message/record schemas shared across the pipeline.

Pydantic models for the SAE J2735 Basic Safety Message (BSM), a grid SCADA node
(IEEE-bus style), a fused multimodal incident, and the grounded diagnostic response.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class AttackLabel(str, Enum):
    BENIGN = "benign"
    POSITION_SPOOF = "position_spoof"      # teleport / GNSS falsification
    SPEED_SPOOF = "speed_spoof"            # kinematically impossible speed
    GHOST = "ghost"                        # fabricated (Sybil) vehicle
    GRID_FDI = "grid_fdi"                  # false-data-injection on SCADA


class BSM(BaseModel):
    """SAE J2735 Basic Safety Message, Part I core data (SI-normalized)."""
    vehicle_id: str
    t: float = Field(..., description="epoch seconds")
    x_m: float = Field(..., description="east position, metres (from local_x)")
    y_m: float = Field(..., description="north position, metres (from local_y)")
    speed_mps: float = Field(..., ge=0)
    accel_mps2: float = 0.0
    heading_deg: float = 0.0
    lane: int = 0
    label: AttackLabel = AttackLabel.BENIGN

    def geokey(self, precision: int = 7) -> str:
        """Cheap deterministic spatial key (grid bucket) for cache/graph nodes."""
        q = 10 ** max(0, precision - 5)
        return f"{int(self.x_m * q)}:{int(self.y_m * q)}"


class SCADANode(BaseModel):
    """Grid bus measurement (IEEE test-case bus)."""
    bus_id: int
    t: float
    p_injection_mw: float = Field(..., description="net real-power injection (gen - load)")
    v_pu: float = Field(1.0, description="voltage magnitude, per-unit")
    theta_rad: float = Field(0.0, description="voltage angle")
    ev_load_mw: float = Field(0.0, description="EV charging load component")
    label: AttackLabel = AttackLabel.BENIGN


class Incident(BaseModel):
    """A fused multimodal incident episode (traffic + grid + scene text)."""
    incident_id: str
    t_start: float
    t_end: float
    geokey: str
    bsms: List[BSM] = Field(default_factory=list)
    scada: List[SCADANode] = Field(default_factory=list)
    scene_text: str = ""
    severity: float = 0.0
    labels: List[AttackLabel] = Field(default_factory=list)

    @property
    def is_attack(self) -> bool:
        return any(l != AttackLabel.BENIGN for l in self.labels)


class RetrievedItem(BaseModel):
    """One retrieved context unit, with its relevance and physics residual."""
    item_id: str
    text: str
    tokens: int
    relevance: float
    physics_residual: float = 0.0
    hops: int = 0


class DiagnosticResponse(BaseModel):
    """Final grounded diagnostic answer."""
    incident_id: str
    diagnosis: str
    root_cause: str
    confidence: float
    physics_residual: float
    violates_physics: bool
    citations: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0
