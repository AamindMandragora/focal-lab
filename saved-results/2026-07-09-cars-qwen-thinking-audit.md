# CARS Qwen Thinking-Mode Audit

Date: 2026-07-09

Purpose: record the CARS/Qwen thinking-mode fix and the affected artifact count.

## Result

Qwen3/Qwen3.5 CARS runs before the 2026-07-09 patch used the default Qwen chat
template, which opens a `<think>` block. The CARS wrapper now passes
`enable_thinking=False` when formatting chat prompts.

Live project checked:

`/Users/aadivyar/Documents/Research/Dynamic CSD Gen/local-finalization/csd-generation`

Workspace copy mirrored:

`/Users/aadivyar/conductor/workspaces/Dynamic CSD Gen/marseille/csd-generation`

## Counts

Audit scope: JSON artifacts under `outputs/**/*.json` whose path contains both
`cars` and one of `qwen3-5`, `qwen35`, or `qwen3.5`.

- Final affected JSON files: 47
- Partial affected JSON files: 1
- Final affected answer rows: 2473
- Partial affected answer rows: 1
- By top output area: `baselines=25`, `controlled_comparison=19`, `local_smoke=4`

Queue status rows checked:

- `logs/cars_spider_bare_prompt_queue_status.tsv`
- Qwen3 CARS status rows: 14
- Exit 0 rows: 5
- Exit 1 rows: 9

Active process caveat:

At audit time, PID `2700` was still running:

```text
python -m synthesis.evaluate.run_legacy_fixed_strategy --strategy cars --dataset spider --eval-model Qwen/Qwen3.5-4B ...
```

That process started before the patch, so it was still using the old loaded
Python code unless restarted.

## Verification

Focused test in the live project:

```text
.venv/bin/pytest tests/test_cars_qwen35_models.py -q
5 passed, 16 warnings
```

Real tokenizer check:

- `Qwen/Qwen3.5-4B` default template contains an open `<think>`.
- `Qwen/Qwen3.5-4B` with `enable_thinking=False` contains `</think>`.
- `Qwen/Qwen2.5-7B-Instruct` default template did not contain `<think>`, so the
  affected audit count was limited to Qwen3/Qwen3.5 CARS artifacts.

## 2026-07-09 Rerun Launch

Killed local old-code CARS process group:

- PID `2700`: `run_legacy_fixed_strategy --strategy cars --dataset spider --eval-model Qwen/Qwen3.5-4B`
- Parent queue PIDs `2622` and `2634`

Verified local process state after `kill -TERM`: PIDs `2622`, `2634`, and `2700`
were gone.

Remote focal check:

```text
ssh -o ConnectTimeout=8 aadivyar@focal ...
ssh: connect to host ggnds-serv-01.cs.illinois.edu port 22: Operation timed out
```

So local old-code runs were killed, but focal could not be verified in this turn.

Verified Spider/Qwen3.5-4B prompt path after the patch:

```text
spider_profile evaluator_default
model_id Qwen/Qwen3.5-4B
enable_thinking False
formatted_has_closed_think True
```

Launched a fresh local Spider/Qwen3.5 CARS rerun queue:

- Queue: `cars_thinking_off_queue`
- Monitor: `cars_thinking_off_queue_monitor`
- Script: `/Users/aadivyar/Documents/Research/Dynamic CSD Gen/local-finalization/csd-generation/.context/run_cars_spider_qwen35_thinking_off_queue.sh`
- Status: `logs/cars_spider_qwen35_thinking_off_queue_status.tsv`
- Driver log: `logs/cars_spider_qwen35_thinking_off_queue_driver.log`
- Output root: `outputs/baselines/cars_thinking_off/`

First active command at launch:

```text
python -m synthesis.evaluate.run_legacy_fixed_strategy --strategy cars --dataset spider --eval-model Qwen/Qwen3.5-2B ... --output-json outputs/baselines/cars_thinking_off/Qwen_Qwen3-5-2B/spider_seed334_test300__tb1__ms600.json
```

## 2026-07-09 Queue Cleanup

Removed stale queued Qwen3.5 CARS work from the remaining local queue:

- `.context/run_results_matrix_remaining_local_queue.sh` now waits for `cars_thinking_off_queue`.
- Its old Qwen3.5 9B CARS block was replaced by a skip message pointing to `cars_thinking_off_queue`.
- The old `cars_bare_prompt_queue_monitor` tmux session was killed.
- `.context/run_cars_spider_bare_prompt_queue.sh` now exits immediately and points to the thinking-off queue, so it cannot be accidentally relaunched.
- `results_matrix_remaining_local_queue` and its monitor were restarted to pick up the edited script.

Post-cleanup active CARS command:

```text
python -m synthesis.evaluate.run_legacy_fixed_strategy --strategy cars --dataset spider --eval-model Qwen/Qwen3.5-2B ... --output-json outputs/baselines/cars_thinking_off/Qwen_Qwen3-5-2B/spider_seed334_test300__tb1__ms600.json
```

Post-cleanup check:

- Active tmux sessions include `cars_thinking_off_queue` and `results_matrix_remaining_local_queue`.
- No active process is writing Qwen3.5 CARS output to `outputs/baselines/cars_bare_prompt/`.
- Shell syntax check passed for all three queue scripts.

## 2026-07-09 NLTK Resource Fix

The restarted CARS run hit this scoring fallback error:

```text
Resource 'punkt_tab' not found.
Attempted to load 'tokenizers/punkt_tab/english/'
```

Installed the missing NLTK resources into the virtualenv search path:

```text
.venv/nltk_data/tokenizers/punkt
.venv/nltk_data/tokenizers/punkt_tab/english
.venv/nltk_data/corpora/stopwords
```

Moved the interrupted pre-fix artifacts aside:

```text
outputs/baselines/cars_thinking_off/_interrupted/spider_seed334_test300__tb1__ms600.partial.json.before_nltk_fix_20260709T065530Z
logs/interrupted/local_cars_thinking_off_qwen35_2b_spider_full.log.before_nltk_fix_20260709T065530Z
```

Restarted the thinking-off queue:

```text
tmux session: cars_thinking_off_queue
active PID: 22975
active output: outputs/baselines/cars_thinking_off/Qwen_Qwen3-5-2B/spider_seed334_test300__tb1__ms600.partial.json
active log: logs/local_cars_thinking_off_qwen35_2b_spider_full.log
```

Fresh-log check after restart:

```text
rg -n "punkt_tab|Resource .*not found|Batch Spider evaluator failed|Traceback|LookupError" logs/local_cars_thinking_off_qwen35_2b_spider_full.log
```

No matches were found. The fresh log had accepted CARS samples and `Total suceeses`
lines after the restart.
