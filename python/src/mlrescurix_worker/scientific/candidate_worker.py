"""Trusted candidate evaluator process; distinct from untrusted script sandbox."""
import json
import hashlib
import sys
from pathlib import Path


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (request.get("cpu_seconds", 120), request.get("cpu_seconds", 120)))
    if sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (request["memory_bytes"], request["memory_bytes"]))
    # RLIMIT_AS is unreliable for scientific libraries on macOS; never claim cgroups.
    import pandas as pd
    from .candidate_pipeline import identity
    from .task_adapters import task_adapter
    from .evidence import atomic_json, seal_result, source_identity, streaming_file_sha256
    try:
        if source_identity(Path(__file__).resolve().parents[4]) != request['binding']['source_sha256']:
            raise ValueError('Worker source differs from bound source')
        if streaming_file_sha256(request['dataset']) != request['binding']['data_sha256']:
            raise ValueError('Worker development bytes differ from bound input')
        frame = pd.read_csv(request["dataset"])
        adapter = task_adapter(request['task_type'])
        adapter.validate(frame, request['folds'], request.get('topology', {}), request['target'])
        if request.get("operation") == "refit":
            import joblib
            model = adapter.refit(request, frame)
            path = Path(request["model_path"])
            joblib.dump(model, path)
            result = {"model_path": str(path), "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        else:
            result = adapter.evaluate(request, frame)
        result = seal_result(result, request['binding'], 'trusted_installed_worker_development_only')
        payload = {"result": result, "payload_identity": identity(result)}
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        payload = {"error": error, "payload_identity": identity({'error': error})}
    payload['request_identity'] = identity(request)
    atomic_json(Path(sys.argv[2]), payload)


if __name__ == "__main__": main()
