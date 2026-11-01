"""Single campaign governance boundary; Rust remains the authoritative dispatch fence.

The existing JSON authority is preserved without persistence migration. Runtime
response stores and pending intents must be retained for operator reconciliation.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from decimal import Decimal
from typing import Any, Dict

class BudgetExhaustionError(RuntimeError):
    """Raised when an operation would exceed campaign or per-run spending caps."""


class CampaignBudgetAuthority:
    """Authoritative, transactional campaign-wide spend authority.

    Guarantees that USD 0.30 campaign ceiling and USD 0.05 per-run ceiling are enforced
    transactionally across all sessions, runs, retries, and uncertain charges.
    """

    @classmethod
    def open_preserved(cls, campaign_dir, policy):
        if not (Path(campaign_dir) / 'campaign_budget_ledger.json').is_file():
            raise FileNotFoundError('Preserved campaign ledger missing; restore and reconcile, never create a new allowance')
        return cls(campaign_dir, policy, require_existing=True)

    def __init__(self, campaign_dir: Path, policy: Dict[str, Any], *, require_existing=False):
        self.require_existing = require_existing
        self.campaign_dir = Path(campaign_dir)
        self.campaign_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.campaign_dir / "campaign_budget_ledger.json"
        self.lock_path = self.campaign_dir / "campaign_budget_ledger.lock"
        self.policy = policy
        self.campaign_ceiling = Decimal(str(policy["campaign_budget"]["provider_usd_max"]))
        self.per_run_ceiling = Decimal(str(policy["provider"]["usd_per_main_live_run_max"]))
        self.contingency = Decimal(str(policy["campaign_budget"].get("contingency_usd", "0.05")))
        self._ensure_ledger_initialized()

    def _ensure_ledger_initialized(self) -> None:
        import fcntl
        with open(self.lock_path, "a+", encoding="utf-8") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                if not self.ledger_path.exists():
                    if self.require_existing:
                        raise FileNotFoundError("Preserved ledger disappeared; reconciliation required")
                    initial_data = {
                        "campaign_ceiling_usd": str(self.campaign_ceiling),
                        "per_run_ceiling_usd": str(self.per_run_ceiling),
                        "contingency_usd": str(self.contingency),
                        "total_billed_usd": "0.00",
                        "total_reserved_usd": "0.00",
                        "total_uncertain_usd": "0.00",
                        "remaining_campaign_usd": str(self.campaign_ceiling),
                        "runs": {},
                        "active_reservations": {},
                    }
                    tmp = self.ledger_path.with_suffix(f".tmp.{os.getpid()}")
                    tmp.write_text(json.dumps(initial_data, indent=2), encoding="utf-8")
                    tmp.replace(self.ledger_path)
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)

    def _transition(self, run_id, request_id, operation, amount, cost_kind="unverified"):
        import fcntl
        amount = Decimal(str(amount))
        if not amount.is_finite() or amount < 0 or (operation == "reserve" and amount == 0):
            raise ValueError("Budget amounts must be finite and nonnegative; reservations must be positive")
        if not run_id or not request_id:
            raise ValueError("Stable run/request identity required")
        with open(self.lock_path, "a+", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(self.ledger_path.read_text())
            records = data.setdefault("request_records", {})
            active = data.setdefault("active_reservations", {})
            old = active.get(request_id) or records.get(request_id)
            if old and old["run_id"] != run_id:
                raise ValueError("Request identity belongs to a different run")
            if operation == "reserve" and old:
                if Decimal(old["amount_usd"]) != amount:
                    raise ValueError("Request reservation changed")
                if old.get("status", "reserved") != "reserved":
                    raise ValueError("Request already settled or ambiguous; reconcile without redispatch")
                return amount
            if operation != "reserve" and not old:
                raise ValueError("Cannot settle an unreserved request")
            if operation != "reserve" and old.get("status") in ("settled", "uncertain"):
                if old["status"] == "settled" and operation == "settle" and Decimal(old["settled_usd"]) == amount and old.get("cost_kind") == cost_kind:
                    return
                if old["status"] == "uncertain" and operation == "uncertain":
                    return
                raise ValueError("Terminal request requires explicit evidence reconciliation")
            run = data.setdefault("runs", {}).setdefault(run_id, {
                "billed_usd": "0.00", "reserved_usd": "0.00", "uncertain_usd": "0.00", "requests": 0})
            ceiling = min(self.campaign_ceiling, Decimal(data["campaign_ceiling_usd"]))
            per_run = min(self.per_run_ceiling, Decimal(data["per_run_ceiling_usd"]))
            def change(total, field, delta):
                data[total] = str(Decimal(data[total]) + delta)
                run[field] = str(Decimal(run[field]) + delta)
            if operation == "reserve":
                committed = sum(Decimal(data[k]) for k in ("total_billed_usd", "total_reserved_usd", "total_uncertain_usd"))
                run_committed = sum(Decimal(run[k]) for k in ("billed_usd", "reserved_usd", "uncertain_usd"))
                if committed + amount > ceiling:
                    raise BudgetExhaustionError("Campaign spending ceiling exhausted")
                if run_committed + amount > per_run:
                    raise BudgetExhaustionError("Per-run spending ceiling exhausted")
                old = {"run_id": run_id, "amount_usd": str(amount), "status": "reserved"}
                active[request_id] = dict(old)
                change("total_reserved_usd", "reserved_usd", amount)
            else:
                reserved = Decimal(old["amount_usd"])
                if operation == "uncertain" and amount != reserved:
                    raise ValueError("Uncertain amount must retain the complete reservation")
                # Estimates cannot release unused reservation; provider charges remain unknown.
                retained = max(amount, reserved) if cost_kind != "provider_verified" else amount
                if operation == "settle" and amount > reserved:
                    # Record an actual overrun rather than hiding known liability.
                    # The controller stops and subsequent admission uses the higher total.
                    old["reservation_exceeded"] = True
                change("total_reserved_usd", "reserved_usd", -reserved)
                active.pop(request_id, None)
                if operation == "settle":
                    change("total_billed_usd", "billed_usd", amount)
                    change("total_uncertain_usd", "uncertain_usd", retained - amount)
                    run["requests"] += 1
                    old.update(status="settled", settled_usd=str(amount), cost_kind=cost_kind,
                               unreconciled_usd=str(retained - amount))
                else:
                    change("total_uncertain_usd", "uncertain_usd", reserved)
                    old.update(status="uncertain", cost_kind="unknown")
            records[request_id] = old
            data["remaining_campaign_usd"] = str(max(Decimal(0), ceiling - sum(
                Decimal(data[k]) for k in ("total_billed_usd", "total_reserved_usd", "total_uncertain_usd"))))
            tmp = self.ledger_path.with_suffix(f".tmp.{os.getpid()}")
            with tmp.open("w") as stream:
                stream.write(json.dumps(data, indent=2) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            tmp.replace(self.ledger_path)
            directory_fd = os.open(self.campaign_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            return amount

    def reserve(self, run_id, request_id, amount_usd):
        return self._transition(run_id, request_id, "reserve", amount_usd)

    def settle(self, run_id, request_id, actual_billed_usd, *, cost_kind="unverified"):
        return self._transition(run_id, request_id, "settle", actual_billed_usd, cost_kind)

    def mark_uncertain(self, run_id, request_id, reserved_usd):
        return self._transition(run_id, request_id, "uncertain", reserved_usd)

    def get_status(self) -> Dict[str, Any]:
        import fcntl
        with open(self.lock_path, "a+", encoding="utf-8") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_SH)
            try:
                return json.loads(self.ledger_path.read_text(encoding="utf-8"))
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)


class GovernanceService:
    """Admission, durable response checkpointing and conservative settlement.

    Authorization/route/content binding and response recovery are delegated only
    to the campaign-bound Rust gateway; this service has no approval authority.
    """
    def reservation(self, session, messages, max_tokens):
        if callable(getattr(session.provider_caller, "reservation", None)):
            return session.provider_caller.reservation(messages, max_tokens)
        if session.provider_caller is not None and session.provider_is_fixture and session.pricing is None: return Decimal("0.005")
        if not session.pricing or not session.pricing.get("verified"):
            raise ValueError("Live dispatch requires verified input/output price ceilings; no stale guessed rates")
        # UTF-8 bytes upper bound token count, plus conservative chat framing allowance.
        input_bound = len(json.dumps(messages, ensure_ascii=False).encode()) + 4096
        for rate in ("input_per_million", "output_per_million"):
            value = Decimal(str(session.pricing[rate]))
            if not value.is_finite() or value < 0: raise ValueError("Verified price ceilings must be finite and nonnegative")
        return max(Decimal("0.000001"), (Decimal(input_bound) * Decimal(str(session.pricing["input_per_million"])) +
            Decimal(max_tokens) * Decimal(str(session.pricing["output_per_million"])))) / Decimal(1000000)

    def dispatch(self, session, messages, max_tokens=2500):
        from .autonomous_loop import RecoveryRequired
        pending = session.state["pending_turn"]
        request_id = pending["request_id"]
        if not session.provider_is_fixture:
            from .rust_gateway import RustGatewayAdapter
            if not isinstance(session.provider_caller, RustGatewayAdapter):
                raise ValueError("Live dispatch requires the campaign-bound Rust gateway adapter")
        amount = session._reservation(messages, max_tokens)
        # Upper-bound context and requested output across all turns, including repair calls.
        requested_bound = len(json.dumps(messages).encode()) + 4096 + max_tokens
        prior_bound = sum(t.get("requested_token_upper_bound", 0) for t in session.state["turns"])
        token_ceiling = getattr(session, "max_run_tokens", None)
        policy = getattr(session, "policy", None) or {}
        if token_ceiling is None:
            cb_tokens = policy.get("campaign_budget", {}).get("max_total_tokens")
            b_tokens = policy.get("budget", {}).get("tokens")
            if cb_tokens is not None:
                token_ceiling = cb_tokens
            elif b_tokens is not None:
                token_ceiling = b_tokens
            else:
                token_ceiling = 5000000
        if prior_bound + requested_bound > token_ceiling:
            raise ValueError(f"Run token admission limit ({token_ceiling}) exceeded")
        pending["requested_token_upper_bound"] = requested_bound
        spent = sum(Decimal(str(t["provider_cost_usd"])) for t in session.state["turns"])
        if spent + amount > session.per_run_budget_usd: raise ValueError("Per-run budget admission rejected")
        status = session.campaign_authority.get_status()
        ceiling_val = getattr(session.campaign_authority, "campaign_ceiling", None)
        if ceiling_val is None:
            cb_max = policy.get("campaign_budget", {}).get("provider_usd_max")
            if cb_max is not None:
                ceiling_val = cb_max
            else:
                ceiling_val = status.get("campaign_ceiling_usd", "100.00")
        campaign_ceiling = Decimal(str(ceiling_val))
        if Decimal(str(status.get("campaign_ceiling_usd", campaign_ceiling))) > campaign_ceiling:
            raise ValueError(f"Campaign authority exceeds authorized ${campaign_ceiling} ceiling")
        session.campaign_authority.reserve(session.session_id, request_id, amount)
        session._event("provider_reserved", request_id=request_id, reserved_usd=str(amount), fixture=session.provider_is_fixture)
        pending.update(phase="provider_dispatching", reserved_usd=str(amount)); session._save_state(session.state)
        try:
            if session.provider_caller is not None:
                stable_call = getattr(session.provider_caller, "call_with_request_id", None)
                res = stable_call(messages, max_tokens, request_id) if callable(stable_call) else session.provider_caller(messages, max_tokens)
            else: res = session._http_provider(messages, max_tokens)
            if "cost_usd" not in res: raise ValueError("Provider must return billed cost or marked conservative cost estimate")
            cost = Decimal(str(res["cost_usd"]))
            if not cost.is_finite() or cost < 0: raise ValueError("Invalid provider cost")
            # Durable response precedes settlement, so an interrupted settlement is reconciled, never resent.
            pending.update(phase="provider_received", provider_receipt=res); session._save_state(session.state)
            session.campaign_authority.settle(session.session_id, request_id, cost) if session.provider_is_fixture else session.campaign_authority.settle(
                session.session_id, request_id, cost, cost_kind="calculated_estimate" if res.get("cost_is_estimate", True) else "unverified")
            pending["phase"] = "provider_settled"; session._save_state(session.state)
            if cost > amount: raise RecoveryRequired("Provider charge exceeded verified reservation; reconcile pricing before continuation")
            session._event("provider_settled", request_id=request_id, cost_usd=str(cost), estimated=res.get("cost_is_estimate", False))
            return res
        except BaseException:
            if pending["phase"] == "provider_dispatching":
                session.campaign_authority.mark_uncertain(session.session_id, request_id, amount)
            raise
