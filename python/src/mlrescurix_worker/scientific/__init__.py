"""M2 scientific worker and standard-library local dashboard package."""

from importlib import import_module

_EXPORTS = {
    "CriticEngine": "critic_engine", "CritiqueReceipt": "critic_engine",
    "critic_audit_candidate": "critic_engine", "FullInferencePipeline": "pipeline",
    "AutonomousAgentLoop": "autonomous_loop", "ProcessTreeSampler": "autonomous_loop",
    "get_controller_rss_bytes": "autonomous_loop",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value

