# Claude Code synthesis provider and GSM-14B live switch

Date: 2026-07-15

## Purpose

Add a canonical Claude Code Max author provider without an API fallback, then move only the unfinished GSM Qwen2.5-14B recovery from AWS Bedrock to that provider at a saved attempt boundary.

## Approved account and cost surface

- Claude account: `aadivya@fermi.ai`
- Provider reported by Claude Code: `claude.ai`, `firstParty`, subscription `max`
- Model and effort: `claude-sonnet-4-6`, `high`
- Credential source: `/home/aadivyar/.claude-csd-synthesis`
- This route uses the existing Claude Max allowance. It does not call the Anthropic API, AWS Bedrock, OpenAI, Gemini, or Vertex as a fallback.
- A previously exposed OpenAI credential in focal shell configuration was not read, copied, printed, or used. Rotation was explicitly deferred by the user on 2026-07-15.

## Provider result

Canonical names:

- `claude`: Claude Code Max through the isolated focal config.
- `claude-bedrock`: AWS Bedrock.
- `anthropic`: direct Anthropic API.
- Deprecated aliases `claude-code` and `bedrock` still work with warnings.

The Claude child process receives a small allowed environment, a fresh temporary home and working directory, no tools, no slash commands, no MCP config, no session persistence, and prompt bytes over standard input. Account preflight requires the exact approved email and Max/first-party fields.

One allowance-only focal validation call returned exactly `CLAUDE_CSD_PROVIDER_OK` with exit code 0.

## Tests and review

- Initial provider suite: `18 failed, 1 passed` before implementation.
- Final focal provider, recovery, checkpoint, and launcher suite: `73 passed, 3 warnings in 12.57s`.
- Python compilation and `git diff --check`: passed.
- Independent judge: `PASS` after checking process cleanup, timeout handling, redaction, no-fallback proof, account checks, diagnostics, and profile mappings.
- A separate live-deployment judge initially found the missing worktree recovery dependency and missing restart claim. After both fixes, it returned `PASS`: worktree/focal feedback-loop hash parity, live claim presence, exit-75 handling, rollback disable behavior, unchanged worker PID, and one post-switch attempt-55 block were all verified.

The three warnings were two existing SWIG deprecation warnings and one expected migration warning for the `sonnet4.6` matrix profile.

## Durable switch boundary

The old Bedrock worker had finished attempts 46 through 54 and had already authored attempt 55. It was stopped during attempt-55 local GPU evaluation, before a new author call.

Checkpoint artifacts on focal:

- `.context/claude_code_resume_0715/gsm14b_before55.json`
  - 52 evaluated records, ending at attempt 54.
  - SHA-256: `1a9f647306d82bf3860466da9bedd13dae22b673564ab7ae14a365076bbbb629`
- `.context/claude_code_resume_0715/gsm14b_attempt55.dfy`
  - Exact already-authored attempt-55 strategy recovered from the live log.
  - SHA-256: `e0733965a6b53408c750f66fbff795e1c766cde35ba387221be0987dbad53446`

The new process uses `--initial-attempt-offset 54 --max-iterations 26`. It therefore evaluates attempt 55 once, authors the first new Claude Code strategy as attempt 56, and retains the original cap of 80.

The one-time checkpoint is permanently claimed at `.context/claude_code_resume_0715/attempt55.claim`. The launcher never removes this directory. Any abnormal service restart or later login therefore exits with status 75 instead of replaying attempt 55 or duplicating later author calls.

## Live state after deployment

- Service: `csd-gsm14b-claude-resume.service`
- Service main PID at verification: `3948047`
- Synthesis PID at verification: `3948057`
- Active GSM-14B synthesis workers: `1`
- Service restarts: `0`
- GPU 2 reached `28886 MiB` and began attempt-55 evaluation.
- `csd-warm-recovery.service`: active, zero restarts.
- `csd-codex-incident-monitor.service`: active, zero restarts.
- The local combined log contains the new `provider=claude` start marker, `AUTHOR MODEL : 'claude-sonnet-4-6' via backend='claude'`, the 52-record history load, and `Attempt 55/80`.

Provider source hashes deployed on focal:

```text
85f37d7465a10baf41c1dccbfd2a3477687c73f513559d497eae8503656e1104  synthesis/generate/generator.py
8bc0aefb4ce37ae9eb1db722bf2285470934a10fc47977c23e2f1cb271140255  synthesis/generate/provider_names.py
871986b063705b74f41ba183390ed7f6b9b093097ccc6029c70a6c44c2f59ac8  synthesis/run_synthesis.py
84d92c9acd257c1b3333920e328db0ca8f36564154470cc5bb2889d33d36b30a  run_all_tests.py
772bde9a93c939454cefc506b8c8602938f4336834af8369a5bbf5b1a700bb3b  synthesis/evaluate/feedback_loop.py
```

## Monitoring

```bash
ssh aadivyar@focal "bash -lc 'systemctl --user status csd-gsm14b-claude-resume.service --no-pager'"
```

```bash
tail -F "/Users/aadivyar/Documents/Research/Dynamic CSD Gen/local-finalization/csd-generation/logs/combined_data_collection.log"
```

## Rollback

The untouched pre-deployment focal files are stored at:

```text
/home/aadivyar/csd-generation/.context/claude_code_provider_backup_20260715T113737Z
```

To stop and prevent future login-time startup of only this recovery run:

```bash
ssh aadivyar@focal "bash -lc 'systemctl --user disable --now csd-gsm14b-claude-resume.service'"
```

Do not restart the older `csd-gsm14b-rendered-resume.service`; it uses the prior Bedrock launcher.
