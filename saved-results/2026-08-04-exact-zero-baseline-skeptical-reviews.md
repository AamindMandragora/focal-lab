# Skeptical review of diverse exact-zero GCD baselines

Date: 2026-08-04

## Decision

Accept these four exact artifacts as genuine zero-score model results. All
decoders functioned and produced varied nonblank outputs; none shows the
blank-output or one-repeated-malformed-output system-failure signature.

## Reviewed artifacts

### smiles-acrylates-qwen25-7b-gcd

- SHA-256: `03a8d326a038d23b898e173842bf59bdde17dbc34aed56dd61e3e28e931076e6`
- Rows: 50
- Nonblank outputs: 50
- Distinct outputs: 19
- Output lengths: 38 to 498 characters
- Outputs at or above the 400-token campaign cap: 49
- Run result: exit 0; artifact saved
- Timing: 901.4808 total generation seconds; 1031.9706 wall-clock seconds
- Failure evidence: varied unclosed-ring, duplicate-ring, and valence failures;
  no traceback, CUDA memory error, or blank-output collapse.

### smiles-isocyanates-qwen25-7b-gcd

- SHA-256: `0a8a553c52f2aa2ef92045f5cbb3dc73217e5a473318a440de7e6467e66cf4b9`
- Rows: 50
- Nonblank outputs: 50
- Distinct outputs: 34
- Output lengths: 59 to 683 characters
- Outputs at or above the 400-token campaign cap: 45
- Run result: exit 0; artifact saved
- Timing: 819.4608 total generation seconds; 829.1132 wall-clock seconds
- Failure evidence: varied unclosed-ring, duplicate-ring, syntax, and valence
  failures; no traceback, CUDA memory error, or blank-output collapse.

### smiles-isocyanates-qwen35-2b-gcd

- SHA-256: `da5330fed79bcb8bf1d22f8085be1c4b7dc93935d054f9008aa370db07f57b17`
- Rows: 50
- Nonblank outputs: 50
- Distinct outputs: 38
- Output lengths: 13 to 599 characters
- Outputs at or above the 400-token campaign cap: 49
- Run result: exit 0; artifact saved
- Timing: 1178.1388 total generation seconds; 1806.1647 wall-clock seconds
- Failure evidence: varied RDKit syntax and duplicate-ring failures; no
  traceback, CUDA memory error, or blank-output collapse.
- Decoder assessment: 49/50 outputs reaching the cap is a severe decoding
  failure, but the varied, fully scored generations do not show a harness fault.

### smiles-acrylates-qwen25-1p5b-gcd sampling-v2

- SHA-256: `5287e4c61cba6e09bc2836557b17d0a4ffb64b444040d16a38aa22d0c6c6d256`
- Supersedes quarantined SHA-256:
  `f0f378de0e02c6d4120154b92531e9fec70d0b6f2d94d88ff8a71f835a0ab587`
- Rows: 50
- Nonblank outputs: 50
- Distinct outputs: 18
- Outputs at the 400-token campaign cap: 49
- Run result: exit 0; artifact saved; no traceback, CUDA memory error, kill,
  or timeout.
- Independent review: PASS by `gpt-5.6-sol`, bound to the replacement SHA.
- Decoder assessment: one output mode appeared 25/50 times and 49/50 outputs
  reached the cap. This is a severe model decoding failure, but 17 other modes
  and one 134-token completion show that the repaired sampling path was not
  frozen or fixed-output.

## Reproduce

From the focal worktree, hash and inspect:

```bash
sha256sum outputs/baselines/exact-zero-repair-20260804/smiles-acrylates/qwen25-7b/gcd.json
sha256sum outputs/baselines/exact-zero-repair-20260804/smiles-isocyanates/qwen25-7b/gcd.json
sha256sum outputs/baselines/exact-zero-repair-20260804/smiles-isocyanates/qwen35-2b/gcd.json
sha256sum outputs/baselines/exact-zero-repair-20260804-gcd-sampling-v2/smiles-acrylates/qwen25-1p5b/gcd.json
rg -n 'Traceback|CUDA out of memory|Saved baseline JSON' logs/exact-zero-repair-20260804/smiles-{acrylates,isocyanates}-qwen25-7b-gcd.log
rg -n 'Traceback|CUDA out of memory|Saved baseline JSON' logs/exact-zero-repair-20260804/smiles-isocyanates-qwen35-2b-gcd.log
rg -n 'Traceback|CUDA out of memory|Killed|timeout|Saved baseline JSON' logs/exact-zero-repair-20260804-gcd-sampling-v2/smiles-acrylates-qwen25-1p5b-gcd.log
```

The monitor consumes the matching entries in
`saved-results/2026-08-04-exact-zero-baseline-acceptances.json`; the versioned
replacement is bound separately in
`saved-results/2026-08-04-exact-zero-baseline-supersessions.json`.
