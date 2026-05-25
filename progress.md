# Experiment progress

_Last updated: 2026-05-25T15:40:12+00:00_

## Status: ✅ RUN ACTIVE  (PID 896681, uptime 03:37:02)

## GPU

```
0, 508 MiB, 40960 MiB, 0 %
1, 34509 MiB, 40960 MiB, 5 %
2, 32986 MiB, 40960 MiB, 80 %
3, 32983 MiB, 40960 MiB, 0 %
```

## Completed cells (since launch_full_matrix_20260520_104938.sh)

| Strategy | Model | Benchmark | Acc | Syntax | N | Saved |
|---|---|---|---|---|---|---|
| crane | Qwen_Qwen2.5_1.5B_Instruct | gsm_symbolic | 38.0% | ? | 50 | 2026-05-25T06:23:18 |
| crane | Qwen_Qwen2.5_1.5B_Instruct | spider | 20.0% | ? | 100 | 2026-05-24T16:47:42 |
| crane | Qwen_Qwen2.5_Coder_7B_Instruct | gsm_symbolic | 54.3% | ? | 70 | 2026-05-24T15:14:04 |
| crane | Qwen_Qwen2.5_Coder_7B_Instruct | gsm_symbolic | 50.0% | ? | 50 | 2026-05-24T23:17:21 |
| crane | Qwen_Qwen2.5_Coder_7B_Instruct | spider | 69.0% | ? | 100 | 2026-05-20T11:45:42 |
| gcd | Qwen_Qwen2.5_1.5B_Instruct | gsm_symbolic | 26.0% | ? | 50 | 2026-05-25T10:34:56 |
| gcd | Qwen_Qwen2.5_Coder_7B_Instruct | gsm_symbolic | 47.6% | ? | 82 | 2026-05-25T11:53:44 |
| gcd | Qwen_Qwen2.5_Coder_7B_Instruct | gsm_symbolic | 46.0% | ? | 50 | 2026-05-24T14:12:37 |
| gcd | Qwen_Qwen2.5_Coder_7B_Instruct | gsm_symbolic | 46.0% | ? | 50 | 2026-05-24T23:12:26 |
| gcd | Qwen_Qwen2.5_Coder_7B_Instruct | gsm_symbolic | 43.8% | ? | 64 | 2026-05-25T11:53:32 |
| gcd | Qwen_Qwen2.5_Coder_7B_Instruct | spider | 37.0% | ? | 100 | 2026-05-20T11:43:22 |
| itergen | Qwen_Qwen2.5_1.5B_Instruct | gsm_symbolic | 20.0% | ? | 50 | 2026-05-25T06:59:18 |
| itergen | Qwen_Qwen2.5_Coder_7B_Instruct | gsm_symbolic | 22.0% | ? | 50 | 2026-05-24T15:22:38 |
| itergen | Qwen_Qwen2.5_Coder_7B_Instruct | gsm_symbolic | 24.0% | ? | 50 | 2026-05-24T23:40:10 |
| itergen | Qwen_Qwen2.5_Coder_7B_Instruct | gsm_symbolic | 24.0% | ? | 50 | 2026-05-25T08:05:02 |
| itergen | Qwen_Qwen2.5_Coder_7B_Instruct | spider | 70.0% | ? | 100 | 2026-05-25T13:56:45 |
| metadecode | Qwen_Qwen2.5_Coder_7B_Instruct | gsm_symbolic | 0.0% | ? | 0 | 2026-05-25T13:54:16 |
| unconstrained | Qwen_Qwen2.5_1.5B_Instruct | gsm_symbolic | 28.0% | ? | 50 | 2026-05-25T10:32:31 |
| unconstrained | Qwen_Qwen2.5_Coder_7B_Instruct | spider | 69.0% | ? | 100 | 2026-05-20T11:40:45 |

## Errors in latest log

Total error/traceback lines: **66**

Unique error types:
```
(EngineCore pid=1906250) ERROR 05-20 11:49:38 [core.py:1136]     raise ValueError(
(EngineCore pid=1906250) ERROR 05-20 11:49:38 [core.py:1136] ValueError: Free memory on device cuda:0 (4.9/39.49 GiB) on startup is less than desired GPU memory utilization (0.85, 33.57 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.
(EngineCore pid=1906250)     raise ValueError(
(EngineCore pid=1906250) ValueError: Free memory on device cuda:0 (4.9/39.49 GiB) on startup is less than desired GPU memory utilization (0.85, 33.57 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.
(EngineCore pid=1907376) ERROR 05-20 11:50:00 [core.py:1136]     raise ValueError(
(EngineCore pid=1907376) ERROR 05-20 11:50:00 [core.py:1136] ValueError: Free memory on device cuda:0 (4.9/39.49 GiB) on startup is less than desired GPU memory utilization (0.85, 33.57 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.
(EngineCore pid=1907376)     raise ValueError(
(EngineCore pid=1907376) ValueError: Free memory on device cuda:0 (4.9/39.49 GiB) on startup is less than desired GPU memory utilization (0.85, 33.57 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.
(EngineCore pid=1908054) ERROR 05-20 11:50:12 [core.py:1136]     raise ValueError(
(EngineCore pid=1908054) ERROR 05-20 11:50:12 [core.py:1136] ValueError: Free memory on device cuda:0 (4.9/39.49 GiB) on startup is less than desired GPU memory utilization (0.55, 21.72 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.
(EngineCore pid=1908054)     raise ValueError(
(EngineCore pid=1908054) ValueError: Free memory on device cuda:0 (4.9/39.49 GiB) on startup is less than desired GPU memory utilization (0.55, 21.72 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.
(EngineCore pid=1909179) ERROR 05-20 11:50:34 [core.py:1136]     raise ValueError(
(EngineCore pid=1909179) ERROR 05-20 11:50:34 [core.py:1136] ValueError: Free memory on device cuda:0 (4.9/39.49 GiB) on startup is less than desired GPU memory utilization (0.55, 21.72 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.
(EngineCore pid=1909179)     raise ValueError(
```

## Latest log tail

```
(EngineCore pid=1930687)     self._init_executor()
(EngineCore pid=1930687)   File "/apps/conda/advayth2/envs/advayth2/lib/python3.12/site-packages/vllm/v1/executor/uniproc_executor.py", line 47, in _init_executor
(EngineCore pid=1930687)     self.driver_worker.init_device()
(EngineCore pid=1930687)   File "/apps/conda/advayth2/envs/advayth2/lib/python3.12/site-packages/vllm/v1/worker/worker_base.py", line 317, in init_device
(EngineCore pid=1930687)     self.worker.init_device()  # type: ignore
(EngineCore pid=1930687)     ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=1930687)   File "/apps/conda/advayth2/envs/advayth2/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=1930687)     return func(*args, **kwargs)
(EngineCore pid=1930687)            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=1930687)   File "/apps/conda/advayth2/envs/advayth2/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py", line 283, in init_device
(EngineCore pid=1930687)     self.requested_memory = request_memory(init_snapshot, self.cache_config)
(EngineCore pid=1930687)                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=1930687)   File "/apps/conda/advayth2/envs/advayth2/lib/python3.12/site-packages/vllm/v1/worker/utils.py", line 413, in request_memory
(EngineCore pid=1930687)     raise ValueError(
(EngineCore pid=1930687) ValueError: Free memory on device cuda:0 (4.9/39.49 GiB) on startup is less than desired GPU memory utilization (0.55, 21.72 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.
[rank0]:[W520 11:57:14.410336981 ProcessGroupNCCL.cpp:1575] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
Retrying vLLM evaluator startup with lower gpu_memory_utilization=0.50
Loading model: Qwen/Qwen2.5-Coder-7B-Instruct on cuda with vLLM...
INFO 05-20 11:57:15 [utils.py:233] non-default args: {'tokenizer': 'Qwen/Qwen2.5-Coder-7B-Instruct', 'trust_remote_code': True, 'max_model_len': 16384, 'enable_prefix_caching': True, 'gpu_memory_utilization': 0.5, 'max_logprobs': -1, 'disable_log_stats': True, 'enforce_eager': True, 'model': 'Qwen/Qwen2.5-Coder-7B-Instruct'}
INFO 05-20 11:57:16 [model.py:555] Resolved architecture: Qwen2ForCausalLM
INFO 05-20 11:57:16 [model.py:1680] Using max model len 16384
WARNING 05-20 11:57:16 [vllm.py:896] Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
WARNING 05-20 11:57:16 [vllm.py:914] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
INFO 05-20 11:57:16 [kernel.py:205] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['vllm_c', 'native'])
INFO 05-20 11:57:16 [vllm.py:1089] Cudagraph is disabled under eager mode
```
