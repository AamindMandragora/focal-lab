# Billing Account of Record — Spider Goal Campaign (Bedrock author spend)

**Date recorded:** 2026-06-23 (overnight autonomous session)
**Why this file exists:** The money rule requires the approved paying account to be written
down in the task's `saved-results` so the spend is auditable. This campaign pays for every
Sonnet-4.6 author call (thinking enabled, effort high) made during synthesis.

## Approved account
- **AWS account: `887730490125` — the UIUC focal lab account** on the `focal` GPU server.
- The user identified this as the UIUC focal lab account and explicitly distinguished it from
  fermi/work and from a personal account (2026-06-22, reaffirmed: *"it should be going to the
  aws key on the focal server which is for my uiuc focal lab NOT fermi"*).
- **Approved scope:** both Spider cells (Qwen2.5-1.5B and 7B), Sonnet-4.6 as the author model,
  for the whole goal campaign (iterate-on-50 → promote to 300-train → final 300-test).
- **NOT a personal account; NOT fermi.** User pre-authorized this spend 2026-06-22.

## How the spend is authenticated (verifiable)
- Backend flag in every run command: `--generation-backend bedrock` → uses
  `AWS_BEARER_TOKEN_BEDROCK` from `~/csd-generation/.env` on focal.
- **`AWS_REGION=us-east-1`** — verified directly from `.env` this session.
- The personal `ANTHROPIC_API_KEY` (also present in `.env`) is **NOT** used by these runs,
  because the backend is `bedrock`, not `anthropic`. The personal key must never be used here.

## Provenance caveat (honest labeling)
- The literal number `887730490125` is **not independently re-derivable this session**: a Bedrock
  bearer token is opaque and does not resolve via `aws sts get-caller-identity`. The number comes
  from prior-session notes plus the user's UIUC-lab identification — recorded, not freshly verified.
- What *is* verified this session: the region (`us-east-1`, from `.env`), the auth path
  (`--generation-backend bedrock` → `AWS_BEARER_TOKEN_BEDROCK`), and that the personal key is unused.

## Live runs charging this account (as of 2026-06-23 22:01 UTC)
- `spider1p5b_iter50_tok0_cold_20260623_220147` (focal pid 2845088, GPU 1)
- `spider7b_iter50_tok0_cold_20260623_220147`   (focal pid 2845089, GPU 2)
- Both COLD, mask ON, author = Sonnet-4.6 via Bedrock. Launchers:
  `run_iter50_tok0_1p5b.sh`, `run_iter50_tok0_7b.sh`.
