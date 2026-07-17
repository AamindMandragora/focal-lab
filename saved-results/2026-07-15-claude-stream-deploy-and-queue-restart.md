# Claude streaming-telemetry deploy + seven-row queue restart

Date: 2026-07-15 (~16:07 UTC start)
For: continuation of the two dead Codex sessions (p0_data collection, p0_research) after the Codex account hit its usage limit (resets 2026-07-22). User approved via question round: "Yes, deploy and restart".

## What this records

The streaming redesign of the Claude Code author provider was deployed to focal, all tests went green, and the user's approved seven synthesis recovery rows were restarted under one controller service.

## Result

- **Tests on focal: 106 passed** (`/apps/conda/aadivyar/envs/csd/bin/python -m pytest tests/providers/test_claude_code_provider.py tests/runtime/claude_recovery/ tests/runtime/incident_repair/ tests/test_synthesis_run_defaults.py -q`, 2026-07-15). An earlier run was 102 passed / 4 failed; the 4 failures were stale copies on focal of the GSM-14B launcher scripts (`.recovery/claude-code-gsm14b/launch_resume_from55*.sh`) and the repo's `deploy/focal/systemd/*.service` files — fixed by copying the worktree versions over.
- **`csd-claude-recovery-queue.service` active** since 16:07:26 UTC, zero restarts at start. One controller supervises all seven rows; GSM-14B is folded in (no separate unit running).
- Live at start: GSM-14B (GPU3), GSM-2B + Spider-7B (GPU0), GSM-4B (GPU1), Spider-4B (GPU2). Queued: GSM-9B, SMILES-isocyanates.
- Legacy units `csd-gsm14b-claude-durable/-helper-resume/-resume` all **inactive** and listed in the queue unit's `Conflicts=` line.
- `csd-codex-incident-monitor.service` restarted with the updated monitor (`--recovery-service csd-claude-recovery-queue.service`); active.

## Verified contract points (from the live command lines, 16:07 UTC)

- Every `run_synthesis` command: `--generation-backend claude --generation-model claude-sonnet-4-6 --claude-expected-account aadivya@fermi.ai` (Claude Max subscription, flat rate — account confirmed live from `/home/aadivyar/.claude-csd-synthesis/.claude.json`: `aadivya@fermi.ai`). No Bedrock.
- `--min-syntax-rate 0.9` on every row (the 0.918… drift from the old manifest is gone; GSM-14B's old 0.85 is also gone).
- New streaming flags on every row: `--claude-timeout-seconds 900 --claude-idle-timeout-seconds 900 --claude-emergency-timeout-seconds 7200 --claude-max-retries 2 --claude-telemetry-dir .context/claude_stream_telemetry --claude-author-lock-file .context/claude_author.lock`.
- `--adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2`, original split files (gsm seed123 49x49, spider seed334 300x300).
- Protected hashes verified intact on focal before start: `prompts.py` sha256 `2e7a0a1748900b072d9c03a25f36836af3a55d2093f4c5ceb32dcb95bad93f9c`, `evaluator.py` `000c8643b4fe1ebd5e4360633b136e934cde6434a6d4d031219b221d6c8b9cb4`.

## Deployment mechanics (how to reproduce/roll back)

- Source of truth deployed: local worktree `/Users/aadivyar/conductor/workspaces/Dynamic CSD Gen/marseille-claude-code-synthesis/csd-generation` → focal `/home/aadivyar/csd-generation` via scp (local-edit-then-scp rule).
- Focal backup of every replaced file: `/home/aadivyar/csd-generation/.context/claude_stream_deploy_backup_20260715T1600Z`.
- Files: `synthesis/generate/generator.py` (stream-json redesign), `synthesis/run_synthesis.py` (`--claude-idle-timeout-seconds`), `scripts/runtime/claude_recovery/*`, `scripts/runtime/incident_repair/*`, `scripts/runtime/run_warm_task_recovery_queue.py`, the five test files, the seven-row manifest `saved-results/2026-07-15-claude-helper-recovery-manifest.json`, systemd units (both to `~/.config/systemd/user/` and to the repo's `deploy/focal/systemd/`), and `.recovery/claude-code-gsm14b/launch_resume_from55*.sh`.
- Manifest rows (all min_syntax_rate 0.9000; SMILES held-out N=100): gsm-qwen25-14b (base 55 / cap 80), gsm-qwen35-2b (10/40), gsm-qwen35-4b (10/40), gsm-qwen35-9b (3/40), spider-qwen35-4b (39/40), spider-qwen25-7b (8/40), smiles-qwen35-9b-isocyanates (1/40). spider-qwen35-9b EXCLUDED (open user decision).

## Open items — all resolved 2026-07-15

1. **Spider-Qwen3.5-9B**: user ruled "Leave excluded" (2026-07-15). The queue stays at seven rows; its existing exhausted-at-40 result stands. Decision is final unless the user reopens it.
2. **GSM-Qwen3.5-4B/9B extension group**: user explicitly APPROVED the `extension_group: "gsm-qwen35-4b-9b"` / `extension_total_cap: 80` in the manifest (2026-07-15, question round). If both rows fail at 40 attempts, `supervise_warm_task_recovery.py` may raise both caps to 80 total. This was the adversarial judge's one FAIL blocker; now cleared by explicit sign-off.
3. **Old combined-log cleanup**: STALE — the judge searched focal (logs/ totals 5.3G, largest file 529M, data disk 449G free) and the Mac tree and found no ~174GB combined log or phone notifier. Nothing to clean up; item closed unless the log turns up elsewhere.
4. **Incident monitor repair engine**: no longer blocked — swapped from Codex CLI (out of credits until 2026-07-22) to Claude Code CLI on the approved Max account `aadivya@fermi.ai` (deployed 2026-07-15, 111 focal tests green, service active).

## Judge verdict: PASS (2026-07-15, second fresh judge)

After the user's two rulings, a fresh adversarial judge re-verified everything on focal and returned **PASS**: protected hashes exact, 111 tests green (fresh run), exact 7-row scope, min-syntax 0.90 everywhere, approved 4B/9B extension implemented correctly (flat 80, only after ALL group rows fail at 40), SMILES N=100, aadivya@fermi.ai Max account with AWS_/BEDROCK/*_API_KEY absent from live process envs, legitimate warm-recovery re-seeds only, mask/bandit/beam flags on all rows, original splits, Claude incident monitor active with account-check-before-stop and no skip-permissions flag, 1:1 GPU mapping, no orphan processes. Flagged non-blocking: monitor.py `_capture_evidence` hardcodes stale unit name `csd-warm-recovery.service` in its diagnostic snapshot (forensics-only; the real stop/restart uses the CLI arg); same stale name as unreachable fallback default; disk 97% used (449G free). The audit brief was copied into this saved-results dir (it previously lived only in /tmp).

## Post-deploy fix: 32k output-token wall (2026-07-16, overnight autonomous run)

Two author calls died with "Claude's response exceeded the 32000 output token maximum": GSM-2B (telemetry claude-20260715T181013Z-2a2bee5311a125f0, 2318 thinking events / 0 text, 3665s) and GSM-14B (claude-20260715T192430Z-b6a260f30912f352, 2549 thinking / 0 text, 8213s). Big authoring prompts (~111KB) spend all 32000 default output tokens on thinking before any strategy text. Spider-7B hit the same wall twice earlier but recovered on retry.

Fix: added `CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000` to `_claude_environment` in `synthesis/generate/generator.py` (the env allowlist is strict, so it must be set in code, not the systemd unit). TDD: red test `test_claude_environment_raises_the_output_token_ceiling` failed on focal pre-fix, then **112 passed** post-fix (same focused suite as the deploy). Pre-fix generator.py backed up to `.context/claude_stream_deploy_backup_20260715T1600Z/generator.py.pre-64k`. Queue restarted ~2026-07-16 00:0x UTC (rows re-seed from last clean attempt by design); both services active. Fair: author-mechanics config only, no grammar/grader/split changes; protected files untouched.

### Follow-up: thinking cap (2026-07-16)

The 64k ceiling took effect but did not cure GSM-2B: a post-fix call (telemetry claude-20260715T221253Z-f0937718f1621100, started 22:12:53 UTC — after the 20:15:53 UTC restart) burned 128000 output tokens across two 64000-token passes, all thinking (2918 thinking events, 0 text), $1.96, 103 min, exit 1. The author thinks without bound on the ~111KB GSM prompts. Fix: added `MAX_THINKING_TOKENS=48000` next to the output cap in `_claude_environment` — caps per-turn thinking below the 64000 output budget so ≥16000 tokens remain for strategy text. Calls that ever succeeded thought under the old 32000 total cap, so 48000 is more generous than any regime that produced a success. TDD red→green: `test_claude_environment_caps_thinking_below_the_output_ceiling` failed pre-fix, **113 passed** post-fix on focal. Backup: `.context/claude_stream_deploy_backup_20260715T1600Z/generator.py.pre-thinking-cap`. Queue restarted again; both services active.

### Follow-up 2: disable adaptive reasoning (2026-07-16 ~02:00 UTC)

The 48000 thinking cap did NOT bind: a post-cap GSM-2B call (claude-20260716T003131Z, 4 API turns, two turns at exactly 64000 output tokens of pure thinking, 4264 thinking events / 0 text, exit 1 with no result event) showed the model still thinking without limit. Docs check (code.claude.com model-config): on Sonnet 4.6, `MAX_THINKING_TOKENS` is only honored as a fixed budget when `CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1` is set — otherwise adaptive reasoning (our `--effort high`) governs thinking. Fix: added `CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1` to `_claude_environment`. TDD red→green, **113 passed** on focal. Backup: `generator.py.pre-adaptive-off` in the same backup dir. Queue restarted ~02:00 UTC (third restart tonight). Verification watcher now reports thinking/text/exit for every completed author call — success = big calls emitting text_events > 0.

## In flight when this was written

- Monitor watching `.context/claude_stream_telemetry` for the first live Claude author call (answers the standing question: were the old 1800s timeouts "model still thinking" or a real stall).
- Adversarial no-drift judge running per `saved-results/2026-07-15-adversarial-drift-audit-brief.md`; deployment is only "complete" when a fresh judge returns PASS with no unresolved current correctness blocker.
