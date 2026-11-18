"""End-to-end integration: ingest real BSM/SCADA -> grounded diagnosis."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physrag.config import Settings
from physrag.etl.synthetic_generator import build_incidents
from physrag.pipeline import PhysRAGPipeline, TokenBucket


@pytest.fixture(scope="module")
def pipe():
    p = PhysRAGPipeline(settings=Settings(plvr_threshold=0.08, token_budget=220,
                                          hnsw_top_k=6))
    p.ingest_many(build_incidents(n_windows=10, attack_fraction=0.5, seed=0))
    return p


def test_pipeline_ingests_real_incidents(pipe):
    assert len(pipe.incidents) >= 6
    assert len(pipe.index) == len(pipe.incidents)
    # the index has graph edges (spatial/temporal)
    assert any(pipe.index.graph.values())


def test_end_to_end_diagnosis_flags_attacks(pipe):
    correct = 0
    for iid, inc in pipe.incidents.items():
        resp = pipe.diagnose(iid, mode="phys_rag")
        assert resp.incident_id == iid
        assert resp.latency_ms >= 0
        if resp.violates_physics == inc.is_attack:
            correct += 1
    assert correct == len(pipe.incidents)        # perfect on this corpus


def test_physrag_grounds_only_on_valid_context(pipe):
    """PhysRAG must not cite physically-invalid (spoofed) evidence."""
    thr = pipe.settings.plvr_threshold
    oracle = {i: pipe.oracle_residual(i) for i in pipe.incidents}
    for iid in pipe.incidents:
        _, context = pipe.diagnose_ex(iid, mode="phys_rag")
        for item in context:
            assert oracle[item.item_id] <= thr, (iid, item.item_id)


def test_base_vlm_is_physics_blind(pipe):
    """Base VLM (no retrieval) cannot flag physics violations -> it leaks them."""
    leaks = 0
    for iid, inc in pipe.incidents.items():
        resp = pipe.diagnose(iid, mode="base_vlm")
        if inc.is_attack and not resp.violates_physics:
            leaks += 1
    assert leaks > 0


def test_token_bucket_rate_limits():
    clock = {"t": 0.0}
    tb = TokenBucket(rate_per_s=10.0, burst=5, clock=lambda: clock["t"])
    assert sum(tb.allow() for _ in range(5)) == 5    # burst
    assert not tb.allow()                            # exhausted
    clock["t"] = 1.0                                 # +10 tokens/s -> refilled
    assert tb.allow()


def test_fastapi_app_if_available():
    try:
        from fastapi.testclient import TestClient
        from physrag.api.main import create_app
    except Exception:
        pytest.skip("fastapi/httpx not installed")
    client = TestClient(create_app())
    assert client.post("/seed", params={"n_windows": 6}).json()["incidents"] >= 1
    m = client.get("/metrics").json()
    assert m["incidents"] >= 1
    # diagnose the first incident
    r = client.post("/diagnose", params={"incident_id": "inc-000", "mode": "phys_rag"})
    assert r.status_code == 200
    assert r.json()["incident_id"] == "inc-000"
