# Bedrock Paid Synthesis Approval

Date: 2026-07-08

Purpose: durable approval record for adding paid Bedrock synthesis jobs to the result-finalization queue.

## User Approval

User approval is explicit for paid Bedrock synthesis.

The user asked to add paid synthesis to the queue and stated that using the AWS Bedrock key on the focal SSH server for synthesis is okay to bill.

## Credential And Account Evidence

- Focal host: `aadivyar@focal`.
- Credential file found on focal: `/home/aadivyar/csd-generation/.env`.
- Non-secret keys found there: `AWS_REGION`, `BEDROCK_OPUS_MODEL`, and `AWS_BEARER_TOKEN_BEDROCK`.
- `aws sts get-caller-identity` could not be used because the `aws` CLI is not installed on focal.
- A Python `boto3` STS check could not identify the account from the repo root because the available credential is a Bedrock bearer token, not standard AWS access keys.
- Project plan account record: `planning/finalize-results-plan.md` records account `887730490125`, lab/UIUC, `us-east-1`, focal `.env` key.

I did not live-verify the account id from STS. The live credential type does not expose that identity through the checks available here. This note records the user's explicit approval to use the focal Bedrock bearer token anyway.

## Cost Scope

Approved queue entries:

- SMILES Qwen3.5 paid metaDecode reruns for the five open cells:
  - 2B chain_extenders
  - 4B acrylates
  - 4B chain_extenders
  - 9B acrylates
  - 9B chain_extenders
- GSM Qwen2.5-14B metaDecode synthesis.
- Spider Qwen2.5-14B metaDecode synthesis.

Rough cost estimate: 7 Bedrock Sonnet-4.6 thinking-high synthesis runs at the prior project estimate of about `$20-60` per run, so about `$140-420` total if every queued run reaches the full 40-iteration budget. Actual cost can be lower if a run accepts early or fails before the full iteration cap.

## Guardrails

- All paid synthesis runs are COLD: no `--initial-strategy-file`.
- Author model: `us.anthropic.claude-sonnet-4-6` through Bedrock.
- Author settings: `--anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized`.
- Max iterations: `40`.
- Queue waits behind the local result-finalization queue before starting focal paid synthesis.
- Queue writes `logs/paid_synthesis_queue_status.tsv`.
- Queue waits for focal GPU `3` to be mostly free before each synthesis/evaluation job.
