"""Persistent eval-pool worker subprocess entrypoint.

Launched by synthesis.scripts.eval_worker_pool.EvalWorkerPool, one process per
idle GPU slot, pinned via CUDA_VISIBLE_DEVICES set by the parent before
Popen. Reads length-prefixed pickle requests from stdin, writes replies to a
DEDICATED response pipe fd (its number passed via the CSD_EVAL_RESP_FD env
var) -- NOT stdout, because vLLM writes plain unstructured text (progress
bars, log lines) to stdout during engine load, which corrupted the framed
protocol when replies shared that channel (observed: a log line got misread
as an 8-byte length header, producing a MemoryError in the parent). Builds
its own Evaluator once (from the "configure" message) and re-uses it --
including its cached vLLM engine (loaded on the first "evaluate" request) --
for every request until told to shut down.

Must be run as `python -m synthesis.scripts.eval_worker_main` (not imported):
vLLM needs its own top-level process to initialize CUDA cleanly, and the
`if __name__ == "__main__":` guard below is required -- a prior timing script
in this workstream crashed without it (vLLM's internal multiprocessing tries
to re-import and re-run the launching module under spawn).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    from synthesis.scripts.eval_worker_pool import recv_msg, send_msg, LOG

    stdin = sys.stdin.buffer
    resp_fd = int(os.environ["CSD_EVAL_RESP_FD"])
    resp_stream = os.fdopen(resp_fd, "wb")

    evaluator = None
    while True:
        req = recv_msg(stdin)
        if req is None:
            print(f"{LOG} worker: stdin closed, exiting", file=sys.stderr, flush=True)
            return

        cmd = req.get("cmd")
        if cmd == "shutdown":
            return

        if cmd == "configure":
            try:
                from synthesis.evaluate.evaluator import Evaluator

                evaluator = Evaluator(**req["config"])
                send_msg(resp_stream, {"ok": True})
            except Exception as exc:
                send_msg(resp_stream, {"ok": False, "error": f"configure failed: {exc}"})
            continue

        if cmd == "evaluate":
            if evaluator is None:
                send_msg(resp_stream, {"ok": False, "error": "worker not configured"})
                continue
            try:
                compiled_module_path = Path(req["compiled_module_path"])
                examples = req["examples"]
                start_index = req["start_index"]
                dataset_len = req["dataset_len"]
                env = evaluator._setup_environment(compiled_module_path)
                logic = evaluator._benchmark_logic()
                run_crane_csd = logic.get_generation_runner()
                smiles_suffix: dict = {}
                results = [
                    evaluator._evaluate_one_example(
                        start_index + i, example, dataset_len, env, logic, run_crane_csd, smiles_suffix
                    )
                    for i, example in enumerate(examples)
                ]
                send_msg(resp_stream, {"ok": True, "results": results})
            except Exception as exc:
                send_msg(resp_stream, {"ok": False, "error": str(exc)})
            continue

        send_msg(resp_stream, {"ok": False, "error": f"unknown cmd {cmd!r}"})


if __name__ == "__main__":
    main()
