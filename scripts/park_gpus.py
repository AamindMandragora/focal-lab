"""Park GPUs by holding a vLLM worker with a small model. Kill with SIGTERM/SIGKILL on the PID."""
import os
import signal
import sys
import time

from vllm import LLM


def main() -> None:
    model = os.environ.get("PARK_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    tp = int(os.environ.get("PARK_TP", "2"))
    util = float(os.environ.get("PARK_GPU_UTIL", "0.85"))
    print(f"[parker] loading model={model} tp={tp} util={util}", flush=True)
    llm = LLM(model=model, tensor_parallel_size=tp, gpu_memory_utilization=util)
    print(f"[parker] loaded; holding GPUs (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')})", flush=True)

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    while True:
        time.sleep(300)


if __name__ == "__main__":
    main()
