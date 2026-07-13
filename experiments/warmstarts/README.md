# Historical strategy bodies

Dafny **strategy method bodies** (or full `MyCSDStrategy` excerpts) saved from
past synthesis attempts. They are retained as provenance and must not seed new
synthesis. The only supported use is pure re-evaluation:

```bash
python3 -m synthesis.run_synthesis \
  --task "..." \
  --dataset gsm_symbolic \
  --initial-strategy-file experiments/warmstarts/<file>.dfy \
  --max-iterations 1 \
  --min-accuracy 0 \
  --min-syntax-rate 0
```

Experiment shell scripts resolve paths via `WARMSTARTS_DIR` in `experiments/scripts/lib.sh`.

## Naming

- **`warmstart_*.dfy`** — historical filenames retained so old run records still resolve (e.g. `warmstart_gsm1p5b_attempt14.dfy`, `warmstart_gsm1p5b_loopint_att9.dfy`, `warmstart_spider7b_a23.dfy`, `warmstart_smiles*.dfy`).
- **`*_body.dfy` / `*_strategy*.dfy`** — descriptive snapshots from specific campaigns (model, split seed, attempt id, date).

Filenames are descriptive; bodies are not verified or compiled until passed
through the normal pipeline. Do not continue synthesis from them.
