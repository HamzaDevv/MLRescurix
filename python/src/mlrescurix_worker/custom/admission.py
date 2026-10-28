"""Resource-Based Admission Controller & Bounded OOM Recovery Manager.

Provides deterministic admission and fault recovery:
1. Admission:
   - Compares PilotResourceReport (peak RSS / GPU memory probe) and ResourceRequestContract
     against local system limits and available accelerators.
   - Evaluates user policy ("local", "kaggle", "auto").
   - Routes eligible workloads to local execution or offloads to Kaggle compute pool.
   - Emits structured `needs_resources` when constraints cannot be satisfied without
     silent capacity reduction.
2. Bounded OOM Policy:
   - Detects CUDA / process memory exhaustion.
   - Discards failed process / attempt.
   - Bisects microbatch size and doubles gradient accumulation to strictly preserve
     effective batch size and mathematical normalization.
   - Enables activation checkpointing if microbatch is already 1.
   - Bounds total retries (<= 3 attempts).
   - Upon retry exhaustion, retains last verified durable checkpoint and cleanly reports
     required hardware without infinite looping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .contracts import PilotResourceReport, ResourceRequestContract

logger = logging.getLogger("mlrescurix.admission")


@dataclass
class AdmissionDecision:
    backend: str  # "local", "kaggle", or "needs_resources"
    admitted: bool
    reason: str
    remedy: Optional[str] = None
    microbatch_size: int = 1
    gradient_accumulation_steps: int = 1
    effective_batch_size: int = 1
    use_amp: bool = False
    amp_dtype: Optional[str] = None
    use_activation_checkpointing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "admitted": self.admitted,
            "reason": self.reason,
            "remedy": self.remedy,
            "microbatch_size": self.microbatch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "use_amp": self.use_amp,
            "amp_dtype": self.amp_dtype,
            "use_activation_checkpointing": self.use_activation_checkpointing,
        }


@dataclass
class OOMRecoveryPlan:
    microbatch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    activation_checkpointing: bool = False
    amp_enabled: bool = False
    retry_count: int = 0
    max_retries: int = 3
    last_durable_checkpoint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "microbatch_size": self.microbatch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "activation_checkpointing": self.activation_checkpointing,
            "amp_enabled": self.amp_enabled,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "last_durable_checkpoint": self.last_durable_checkpoint,
        }


def feature_detect_amp() -> Dict[str, Any]:
    """Probes runtime environment for Automatic Mixed Precision (AMP) support."""
    try:
        import torch

        if torch.cuda.is_available():
            bf16_supported = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
            return {
                "available": True,
                "device": "cuda",
                "dtype": "bfloat16" if bf16_supported else "float16",
                "scaler_required": not bf16_supported,
            }
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return {
                "available": True,
                "device": "mps",
                "dtype": "float16",
                "scaler_required": False,
            }
        # CPU bfloat16 support
        return {
            "available": True,
            "device": "cpu",
            "dtype": "bfloat16",
            "scaler_required": False,
        }
    except Exception:
        return {
            "available": False,
            "device": "cpu",
            "dtype": None,
            "scaler_required": False,
        }


class OOMRecoveryPolicy:
    """Manages state transitions, microbatch bisection, and checkpoint replay on OOM."""

    @staticmethod
    def handle_oom(
        plan: OOMRecoveryPlan, exception: Optional[BaseException] = None
    ) -> Optional[OOMRecoveryPlan]:
        """
        Determines the next recovery configuration when an OOM occurs.

        Strict invariants:
        - Never silently shorten sequence length, alter loss, or drop layers.
        - Preserve effective_batch_size = microbatch_size * gradient_accumulation_steps.
        - Stop cleanly with None when max_retries reached or parameters cannot be reduced further.
        """
        if plan.retry_count >= plan.max_retries:
            logger.warning(
                "OOM recovery attempts exhausted (%d >= %d); stopping cleanly",
                plan.retry_count,
                plan.max_retries,
            )
            return None

        # Strategy 1: Bisect microbatch size and double gradient accumulation
        if plan.microbatch_size > 1:
            if (plan.microbatch_size <= 0 or plan.gradient_accumulation_steps <= 0
                    or plan.microbatch_size * plan.gradient_accumulation_steps != plan.effective_batch_size):
                raise ValueError("Invalid recovery plan: effective batch must equal microbatch times accumulation")
            new_micro = plan.microbatch_size // 2
            while plan.effective_batch_size % new_micro:
                new_micro -= 1
            new_accum = plan.effective_batch_size // new_micro
            logger.info(
                "OOM recovery attempt %d: bisecting microbatch from %d to %d (accumulation: %d -> %d)",
                plan.retry_count + 1,
                plan.microbatch_size,
                new_micro,
                plan.gradient_accumulation_steps,
                new_accum,
            )
            return OOMRecoveryPlan(
                microbatch_size=new_micro,
                gradient_accumulation_steps=new_accum,
                effective_batch_size=plan.effective_batch_size,
                activation_checkpointing=plan.activation_checkpointing,
                amp_enabled=plan.amp_enabled,
                retry_count=plan.retry_count + 1,
                max_retries=plan.max_retries,
                last_durable_checkpoint=plan.last_durable_checkpoint,
            )

        # Strategy 2: If microbatch is already 1, enable activation checkpointing
        if not plan.activation_checkpointing:
            logger.info(
                "OOM recovery attempt %d: microbatch is 1; activating gradient checkpointing",
                plan.retry_count + 1,
            )
            return OOMRecoveryPlan(
                microbatch_size=1,
                gradient_accumulation_steps=plan.effective_batch_size,
                effective_batch_size=plan.effective_batch_size,
                activation_checkpointing=True,
                amp_enabled=plan.amp_enabled,
                retry_count=plan.retry_count + 1,
                max_retries=plan.max_retries,
                last_durable_checkpoint=plan.last_durable_checkpoint,
            )

        # Strategy 3: Both microbatch=1 and activation checkpointing are active; cannot reduce further
        logger.warning(
            "OOM recovery cannot reduce memory further: microbatch is 1 and activation checkpointing is active."
        )
        return None


class AdmissionController:
    """Evaluates resource requests and pilot reports to admit to local or remote backends."""

    def __init__(
        self,
        local_memory_limit_bytes: int = 4 * 1024 * 1024 * 1024,  # 4 GB default local limit
        kaggle_backend: Optional[Any] = None,
    ):
        self.local_memory_limit_bytes = local_memory_limit_bytes
        self.kaggle_backend = kaggle_backend

    def admit(
        self,
        request: ResourceRequestContract,
        pilot_report: Optional[PilotResourceReport] = None,
        user_override: Optional[str] = None,
    ) -> AdmissionDecision:
        """Determines whether to admit locally, offload to Kaggle, or raise needs_resources."""
        override = (user_override or "").strip().lower()
        amp_info = feature_detect_amp()

        # Effective batch calculation
        eff_batch = 16
        if pilot_report and hasattr(pilot_report, "microbatch_size") and pilot_report.microbatch_size:
            eff_batch = pilot_report.microbatch_size

        # 1. User Override: Kaggle
        if override == "kaggle":
            if not self.kaggle_backend:
                return AdmissionDecision(
                    backend="needs_resources",
                    admitted=False,
                    reason="Kaggle backend requested via override but not configured in admission controller.",
                    remedy="Configure Kaggle backend with registered credentials or remove user override.",
                )
            try:
                preflight = self.kaggle_backend.preflight(request, {})
                if preflight.get("status") == "ok":
                    return AdmissionDecision(
                        backend="kaggle",
                        admitted=True,
                        reason="Admitted to Kaggle compute pool via explicit user override.",
                        microbatch_size=eff_batch,
                        gradient_accumulation_steps=1,
                        effective_batch_size=eff_batch,
                        use_amp=amp_info["available"],
                        amp_dtype=amp_info["dtype"],
                    )
                else:
                    return AdmissionDecision(
                        backend="needs_resources",
                        admitted=False,
                        reason=f"Kaggle preflight failed: {preflight.get('reason', 'Quota or hardware unavailable')}",
                        remedy="Wait for Kaggle quota reset or reduce required compute duration.",
                    )
            except Exception as exc:
                return AdmissionDecision(
                    backend="needs_resources",
                    admitted=False,
                    reason=f"Kaggle backend preflight exception: {exc}",
                    remedy="Verify Kaggle credentials and network access.",
                )

        # 2. User Override: Local
        if override == "local":
            if request.accelerator_type.lower() == "cuda":
                try:
                    import torch
                    if not torch.cuda.is_available():
                        return AdmissionDecision(
                            backend="needs_resources",
                            admitted=False,
                            reason="Local execution forced by user override, but CUDA accelerator is unavailable on host. Hardware cannot satisfy request without model modification.",
                            remedy="Allow offloading by removing override or setting override to 'kaggle'.",
                        )
                except ImportError:
                    return AdmissionDecision(
                        backend="needs_resources",
                        admitted=False,
                        reason="Local execution forced by user override, but PyTorch is not installed.",
                        remedy="Install torch or remove local override.",
                    )

            if pilot_report and pilot_report.peak_memory_bytes > self.local_memory_limit_bytes:
                return AdmissionDecision(
                    backend="needs_resources",
                    admitted=False,
                    reason=f"Local execution forced by user override, but pilot peak memory ({pilot_report.peak_memory_bytes // (1024*1024)}MB) exceeds local limit ({self.local_memory_limit_bytes // (1024*1024)}MB).",
                    remedy="Increase local_memory_limit_bytes or switch override to 'kaggle'.",
                )

            return AdmissionDecision(
                backend="local",
                admitted=True,
                reason="Admitted to local execution via explicit user override.",
                microbatch_size=eff_batch,
                gradient_accumulation_steps=1,
                effective_batch_size=eff_batch,
                use_amp=amp_info["available"],
                amp_dtype=amp_info["dtype"],
            )

        # 3. Automatic Resource Admission ("auto" or default)
        needs_remote = False
        offload_reason = ""

        # Check accelerator requirement
        if request.accelerator_type.lower() in ("cuda", "gpu"):
            has_cuda = False
            try:
                import torch
                has_cuda = torch.cuda.is_available()
            except ImportError:
                pass
            if not has_cuda:
                needs_remote = True
                offload_reason = f"Workload requires {request.accelerator_type.upper()} accelerator unavailable on host"

        # Check peak memory requirement from pilot probe
        if pilot_report and pilot_report.peak_memory_bytes > self.local_memory_limit_bytes:
            needs_remote = True
            offload_reason = (
                f"Pilot memory probe ({pilot_report.peak_memory_bytes // (1024*1024)}MB) "
                f"exceeds local memory limit ({self.local_memory_limit_bytes // (1024*1024)}MB)"
            )

        if needs_remote:
            if self.kaggle_backend:
                try:
                    preflight = self.kaggle_backend.preflight(request, {})
                    if preflight.get("status") == "ok":
                        return AdmissionDecision(
                            backend="kaggle",
                            admitted=True,
                            reason=f"Offloaded to Kaggle compute pool: {offload_reason}",
                            microbatch_size=eff_batch,
                            gradient_accumulation_steps=1,
                            effective_batch_size=eff_batch,
                            use_amp=True,
                            amp_dtype="float16",
                        )
                    else:
                        return AdmissionDecision(
                            backend="needs_resources",
                            admitted=False,
                            reason=f"{offload_reason}; Kaggle preflight rejected: {preflight.get('reason')}",
                            remedy="Check Kaggle quota or adjust model microbatch size.",
                        )
                except Exception as exc:
                    return AdmissionDecision(
                        backend="needs_resources",
                        admitted=False,
                        reason=f"{offload_reason}; Kaggle backend error: {exc}",
                        remedy="Resolve Kaggle credentials or quota error.",
                    )
            else:
                return AdmissionDecision(
                    backend="needs_resources",
                    admitted=False,
                    reason=f"{offload_reason}; no remote execution backend configured.",
                    remedy="Register KaggleAccountPool and KaggleExecutionBackend to enable offloading.",
                )

        # Workload fits within local boundaries
        return AdmissionDecision(
            backend="local",
            admitted=True,
            reason="Resource request fits within local hardware boundaries.",
            microbatch_size=eff_batch,
            gradient_accumulation_steps=1,
            effective_batch_size=eff_batch,
            use_amp=amp_info["available"],
            amp_dtype=amp_info["dtype"],
        )
