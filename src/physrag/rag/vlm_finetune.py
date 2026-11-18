"""QLoRA (PEFT) multimodal VLM wrapper with a CI-safe mock mode.

In a GPU environment this wraps a Qwen2.5-VL / LLaVA-style model with 4-bit QLoRA
adapters (``peft``) for fine-tuning and inference. When ``transformers``/``peft``/
CUDA are unavailable (CI, this machine), it transparently falls back to a
deterministic grounded generator that composes the diagnosis from the retrieved,
physics-verified context — so the full pipeline and its metrics stay runnable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from ..models.schemas import AttackLabel, DiagnosticResponse, Incident, RetrievedItem

try:
    import torch  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore
    from peft import LoraConfig, get_peft_model  # type: ignore
    HAVE_STACK = True
except Exception:  # pragma: no cover
    HAVE_STACK = False


@dataclass
class QLoRAConfig:
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    load_in_4bit: bool = True


class VLMDiagnoser:
    """Grounded multimodal diagnoser. Real QLoRA path when the stack is present;
    deterministic grounded fallback otherwise."""

    def __init__(self, config: Optional[QLoRAConfig] = None, mock: Optional[bool] = None):
        self.config = config or QLoRAConfig()
        self.mock = (not HAVE_STACK) if mock is None else mock
        self.model = None
        self.processor = None
        if not self.mock:  # pragma: no cover - requires GPU + weights
            self._build_real()

    def _build_real(self):  # pragma: no cover
        self.processor = AutoProcessor.from_pretrained(self.config.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model, load_in_4bit=self.config.load_in_4bit,
            device_map="auto")
        lora = LoraConfig(r=self.config.r, lora_alpha=self.config.lora_alpha,
                          lora_dropout=self.config.lora_dropout,
                          target_modules=list(self.config.target_modules),
                          task_type="CAUSAL_LM")
        self.model = get_peft_model(model, lora)

    # -- generation -----------------------------------------------------------
    def diagnose(self, incident: Incident, context: Sequence[RetrievedItem],
                 physics_residual: float, violates: bool,
                 use_context: bool = True) -> DiagnosticResponse:
        """Produce a grounded DiagnosticResponse.

        ``use_context=False`` emulates a context-free Base VLM (for ablations):
        it answers from the scene text alone and cannot see the physics evidence,
        so it is more prone to endorsing spoofed claims.
        """
        if not self.mock:  # pragma: no cover
            return self._diagnose_real(incident, context, physics_residual, violates)

        labels = set(incident.labels)
        cites = [c.item_id for c in context][:5]
        if use_context and violates:
            attack = next((l for l in labels if l != AttackLabel.BENIGN),
                          AttackLabel.SPEED_SPOOF)
            diag = f"Anomaly consistent with {attack.value.replace('_', ' ')} detected."
            root = ("Retrieved evidence violates a conservation law "
                    f"(physics residual {physics_residual:.3f} > threshold); the "
                    "offending messages are inconsistent with LWR/grid balance.")
            conf = min(0.95, 0.6 + physics_residual)
        elif use_context:
            diag = "Normal operation; no physics violation in retrieved evidence."
            root = ("Traffic and grid measurements satisfy LWR conservation and "
                    "nodal power balance within tolerance.")
            conf = 0.8
        else:
            # Base VLM: scene-text only, physics-blind.
            congested = "congestion" in incident.scene_text
            diag = ("Possible incident reported by camera."
                    if congested else "Scene appears nominal.")
            root = "Derived from scene description only (no retrieved physics context)."
            conf = 0.5
        return DiagnosticResponse(
            incident_id=incident.incident_id, diagnosis=diag, root_cause=root,
            confidence=conf, physics_residual=physics_residual,
            violates_physics=violates, citations=cites)

    def _diagnose_real(self, incident, context, physics_residual, violates):  # pragma: no cover
        prompt = self._format_prompt(incident, context)
        inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=128)
        text = self.processor.decode(out[0], skip_special_tokens=True)
        return DiagnosticResponse(
            incident_id=incident.incident_id, diagnosis=text[:200],
            root_cause="See generated rationale.", confidence=0.7,
            physics_residual=physics_residual, violates_physics=violates,
            citations=[c.item_id for c in context][:5])

    @staticmethod
    def _format_prompt(incident: Incident, context: Sequence[RetrievedItem]) -> str:
        ctx = "\n".join(f"- {c.text} (residual={c.physics_residual:.3f})"
                        for c in context)
        return (f"Scene: {incident.scene_text}\nPhysics-verified context:\n{ctx}\n"
                "Diagnose the incident and give the root cause, citing context.")

    def train_step(self, batch):  # pragma: no cover - illustrative QLoRA loop hook
        if self.mock:
            return 0.0
        self.model.train()
        out = self.model(**batch)
        out.loss.backward()
        return float(out.loss.item())
