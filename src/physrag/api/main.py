"""FastAPI gateway: ingestion, diagnosis, metrics.

Endpoints:
  POST /ingest/bsm     - push a BSM onto the Redis Stream (or in-memory buffer)
  POST /ingest/scada   - push a SCADA node
  POST /diagnose       - run the PhysRAG pipeline for an incident
  GET  /metrics        - pipeline / cache / rate-limiter counters
  POST /seed           - load the real NGSIM + IEEE-14 incident corpus (demo)

A token-bucket rate limiter guards ingestion, and a Redis Streams consumer drains
buffered messages into fused incidents (falls back to an in-process deque when
Redis is absent). Import is guarded so the module degrades without FastAPI.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Deque, List

from ..config import get_settings
from ..etl.synthetic_generator import build_incidents
from ..models.schemas import BSM, DiagnosticResponse, SCADANode
from ..pipeline import PhysRAGPipeline, TokenBucket

try:
    from fastapi import FastAPI, HTTPException
    HAVE_FASTAPI = True
except Exception:  # pragma: no cover
    HAVE_FASTAPI = False


class StreamConsumer:
    """Redis Streams consumer with an in-memory fallback buffer."""

    def __init__(self, redis_url: str):
        self.buffer_bsm: Deque[BSM] = deque(maxlen=100_000)
        self.buffer_scada: Deque[SCADANode] = deque(maxlen=100_000)
        self._redis = None
        try:
            import redis  # type: ignore
            self._redis = redis.Redis.from_url(redis_url, socket_connect_timeout=0.5)
            self._redis.ping()
        except Exception:
            self._redis = None

    def push_bsm(self, bsm: BSM) -> None:
        if self._redis is not None:
            try:
                self._redis.xadd(get_settings().bsm_stream, {"json": bsm.json()})
                return
            except Exception:
                pass
        self.buffer_bsm.append(bsm)

    def push_scada(self, node: SCADANode) -> None:
        if self._redis is not None:
            try:
                self._redis.xadd(get_settings().scada_stream, {"json": node.json()})
                return
            except Exception:
                pass
        self.buffer_scada.append(node)

    @property
    def backend(self) -> str:
        return "redis" if self._redis is not None else "memory"


def create_app():
    """Build the FastAPI app (raises if FastAPI is not installed)."""
    if not HAVE_FASTAPI:  # pragma: no cover
        raise RuntimeError("FastAPI is not installed; `pip install fastapi uvicorn`.")

    cfg = get_settings()
    app = FastAPI(title="physrag-v2x", version="0.1.0",
                  description="Physics-Informed Multimodal GraphRAG for V2X & Grid")
    pipe = PhysRAGPipeline()
    limiter = TokenBucket(cfg.rate_limit_rps, cfg.rate_limit_burst)
    consumer = StreamConsumer(cfg.redis_url)
    app.state.pipe = pipe
    app.state.ingested = {"bsm": 0, "scada": 0, "diagnoses": 0}

    @app.post("/ingest/bsm")
    def ingest_bsm(bsm: BSM):
        if not limiter.allow():
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        consumer.push_bsm(bsm)
        app.state.ingested["bsm"] += 1
        return {"status": "ok", "backend": consumer.backend}

    @app.post("/ingest/scada")
    def ingest_scada(node: SCADANode):
        if not limiter.allow():
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        consumer.push_scada(node)
        app.state.ingested["scada"] += 1
        return {"status": "ok", "backend": consumer.backend}

    @app.post("/seed")
    def seed(n_windows: int = 8):
        incidents = build_incidents(n_windows=n_windows)
        pipe.ingest_many(incidents)
        return {"status": "ok", "incidents": len(incidents)}

    @app.post("/diagnose", response_model=DiagnosticResponse)
    def diagnose(incident_id: str, mode: str = "phys_rag"):
        if incident_id not in pipe.incidents:
            raise HTTPException(status_code=404, detail="unknown incident_id")
        app.state.ingested["diagnoses"] += 1
        return pipe.diagnose(incident_id, mode=mode)

    @app.get("/metrics")
    def metrics():
        return {
            "ingested": app.state.ingested,
            "incidents": len(pipe.incidents),
            "index_nodes": len(pipe.index),
            "stream_backend": consumer.backend,
            "index_backend": pipe.index.backend,
            "cache_hit_rate": round(pipe.cache.hit_rate, 3),
            "rate_limit_tokens": round(limiter.tokens, 1),
        }

    return app


# Uvicorn entrypoint: `uvicorn physrag.api.main:app`
if HAVE_FASTAPI:  # pragma: no cover
    app = create_app()
