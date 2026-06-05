# Output artifacts

All experiment artifacts follow **model → benchmark → strategy** grouping.

## Baselines

```
outputs/baselines/<model_slug>/<benchmark_key>/<strategy>/tb<token_budget>__ms<max_steps>[suffix].json
```

Examples:

- `outputs/baselines/Qwen_Qwen2.5_Coder_7B_Instruct/gsm_symbolic/crane/tb1__ms900.json`
- `outputs/baselines/Qwen_Qwen2.5_Coder_7B_Instruct/smiles__class_acrylates/rs/tb1__ms900__rs200.json`
- `outputs/baselines/Qwen_Qwen2.5_Coder_7B_Instruct/gsm_symbolic/metadecode/gen_gemini__iter40__tb1__ms900.json`

`<benchmark_key>` is `gsm_symbolic`, `spider`, or `smiles__class_<name>`.

## Generated synthesis runs

```
outputs/generated/<model_slug>/<benchmark_key>/<strategy>/<output_name>_<run_id>/
  dafny/
  python/
  results/
```

`latest_run.txt` and the `latest` symlink stay at `outputs/generated/`.

## Logs

```
logs/<model_slug>/<benchmark_key>/<strategy>/<output_name>_<run_id>/
```

Path helpers: `synthesis/project_paths.py`.
