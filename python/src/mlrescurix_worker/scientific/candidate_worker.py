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
    from .candidate_pipeline import evaluate_candidate
    try:
        frame = pd.read_csv(request["dataset"])
        if request.get("operation") == "refit":
            import joblib
            from .candidate_pipeline import CandidatePipeline
            model = CandidatePipeline(request["spec"], request["target"], request["task_type"]).fit(frame)
            path = Path(request["model_path"])
            joblib.dump(model, path)
            result = {"model_path": str(path), "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        else:
            result = evaluate_candidate(request["spec"], frame, request["target"],
                request["task_type"], request["folds"], request["metric"], request.get("topology"))
        payload = {"result": result}
    except Exception as exc:
        payload = {"error": f"{type(exc).__name__}: {exc}"}
    Path(sys.argv[2]).write_text(json.dumps(payload, allow_nan=False))


if __name__ == "__main__": main()
