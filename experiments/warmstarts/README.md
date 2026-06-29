# Warm-start strategy bodies

Dafny **strategy method bodies** (or full `MyCSDStrategy` excerpts) saved from promising synthesis attempts. Use with manual runs only:

```bash
python -m synthesis.run_synthesis \
  --task "..." \
  --dataset gsm_symbolic \
  --initial-strategy-file experiments/warmstarts/<file>.dfy \
  ...
```

Experiment shell scripts resolve paths via `WARMSTARTS_DIR` in `experiments/scripts/lib.sh`.

## Naming

- **`warmstart_*.dfy`** — canonical warm-start bodies restored for GSM/SMILES/Spider resume scripts (e.g. `warmstart_gsm1p5b_attempt14.dfy`, `warmstart_gsm1p5b_loopint_att9.dfy`, `warmstart_spider7b_a23.dfy`, `warmstart_smiles*.dfy`).
- **`*_body.dfy` / `*_strategy*.dfy`** — descriptive snapshots from specific campaigns (model, split seed, attempt id, date).

Filenames are descriptive; bodies are not verified or compiled until passed through the normal pipeline.
