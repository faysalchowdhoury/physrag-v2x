"""Central configuration (Pydantic settings).

All tunables live here so services, experiments and tests share one source of truth.
Values can be overridden by environment variables prefixed ``PHYSRAG_``.
"""
from __future__ import annotations

from functools import lru_cache

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field
    _HAVE_CONFIGDICT = True
except Exception:  # pragma: no cover - pydantic v1 fallback
    from pydantic import BaseSettings, Field  # type: ignore
    SettingsConfigDict = None  # type: ignore
    _HAVE_CONFIGDICT = False


class Settings(BaseSettings):
    # --- infrastructure ---
    redis_url: str = Field("redis://localhost:6379/0", description="Redis Stack URL")
    bsm_stream: str = "stream:bsm"
    scada_stream: str = "stream:scada"

    # --- retrieval / context packing ---
    token_budget: int = Field(2048, description="max context tokens the packer may use")
    hnsw_top_k: int = Field(32, description="first-stage vector candidates")
    graph_hops: int = Field(2, description="bounded k-hop causal expansion")
    embed_dim: int = Field(64, description="embedding dimensionality (mock encoder)")

    # --- physics ---
    physics_lambda: float = Field(4.0, description="PINN residual penalty weight in re-rank")
    plvr_threshold: float = Field(0.08, description="residual above which a claim violates physics")
    free_flow_speed_fps: float = Field(88.0, description="LWR v_f (ft/s) ~60 mph")
    jam_density_vpm: float = Field(220.0, description="LWR jam density (veh/mile/lane)")

    # --- spatial / caching ---
    geohash_precision: int = Field(7, description="geohash precision for spatial keys")
    cache_capacity: int = Field(4096, description="semantic/geohash cache entries")
    cache_ttl_s: float = Field(30.0, description="cache entry TTL seconds")

    # --- api ---
    rate_limit_rps: float = Field(50.0, description="token-bucket refill rate (req/s)")
    rate_limit_burst: int = Field(100, description="token-bucket capacity")

    # --- data ---
    ngsim_sample: str = Field("data/ngsim_i80_sample.csv.gz",
                              description="committed real NGSIM fixture (I-80)")

    if _HAVE_CONFIGDICT:
        model_config = SettingsConfigDict(env_prefix="PHYSRAG_", case_sensitive=False,
                                          extra="ignore")
    else:  # pragma: no cover - pydantic v1
        class Config:
            env_prefix = "PHYSRAG_"
            case_sensitive = False


@lru_cache
def get_settings() -> "Settings":
    return Settings()
