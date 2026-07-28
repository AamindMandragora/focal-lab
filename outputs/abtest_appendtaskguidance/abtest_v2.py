"""One-off A/B test: re-evaluate the same compiled strategy with 3 guidance strings.

Bypasses reevaluate_compiled_csd.py because that script doesn't expose --gsm-source-dir.
"""
import json, os, sys, time
from pathlib import Path


def main():
    sys.path.insert(0, '/home/aadivyar/csd-generation')
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')

    from synthesis.evaluate.evaluator import Evaluator
    from synthesis.project_defaults import default_gsm_source_dir

    OUT = Path('/home/aadivyar/csd-generation/outputs/abtest_appendtaskguidance')
    ARMS = ['A_original', 'B_empty', 'C_antifmt']

    gsm_dir = default_gsm_source_dir()
    print(f'gsm_source_dir = {gsm_dir}', flush=True)

    ev = Evaluator(
        dataset_name='gsm_symbolic',
        model_name='Qwen/Qwen2.5-Coder-1.5B-Instruct',
        backend='vllm',
        device='cuda',
        sample_size=25,
        max_steps=900,
        step_token_budget=1,
        vllm_gpu_memory_utilization=0.3,
        vllm_tensor_parallel_size=1,
        vllm_max_model_len=16384,
        vllm_enforce_eager=True,
        gsm_source_dir=str(gsm_dir),
    )

    try:
        for arm in ARMS:
            compiled = OUT / arm / 'GeneratedCSD.py'
            assert compiled.is_file(), f'missing {compiled}'
            print(f'\n=== START {arm} {time.strftime("%H:%M:%S")} ===', flush=True)
            res = ev.evaluate_sample(compiled, sample_size=25)
            outpath = OUT / arm / 'result_v2.json'
            payload = {
                'arm': arm,
                'accuracy': res.accuracy,
                'syntax_rate': res.syntax_rate,
                'contains_delimiters': res.contains_delimiters,
                'num_examples': res.num_examples,
                'num_correct': res.num_correct,
                'task_guidance': res.task_guidance,
                'total_time_seconds': res.total_time_seconds,
                'samples': [
                    {
                        'question': s.get('question'),
                        'expected': s.get('expected'),
                        'actual': s.get('actual'),
                        'full_output': s.get('full_output'),
                        'is_correct': s.get('is_correct'),
                        'is_syntax_valid': s.get('is_syntax_valid'),
                        'contains_delimiters': s.get('contains_delimiters'),
                        'task_guidance': s.get('task_guidance'),
                        'token_count': s.get('token_count'),
                    }
                    for s in (res.sample_outputs or [])
                ],
            }
            outpath.write_text(json.dumps(payload, indent=2))
            print(f'=== END {arm} acc={res.accuracy} syn={res.syntax_rate} ===', flush=True)
    finally:
        ev.unload_runtime()
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
