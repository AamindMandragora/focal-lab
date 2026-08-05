# Corrected full-baseline launch approval

Date: 2026-08-05

Purpose: record the independent final review that authorizes the corrected
cold-synthesis queue after the exact-zero baseline repair campaign.

## Bound artifacts

- Git commit: `6766ebd397f4bcdbd4cc3332e1051a9efd6258a6`
- Corrected evidence SHA-256:
  `57392e149cea23efe6f596b921c6cd3d74ae7519e7c286393adadc9bdb579ab7`
- Queue manifest SHA-256:
  `c5f5683311ecf8f03211fb4fd8572b92e472dc5475367afe72413eb31098aa18`
- Recovery history SHA-256:
  `7a3daf63b69ea79ab0cc4af18ae014cc8c3b49a43a836643824a560e1addc88a`

## Independent result

Reviewer: `gpt-5.6-sol`

Decision: approved.

The reviewer independently verified:

- all 31 exact-zero selections and their source/replacement hashes, with
  counts `15 repair-v1 + 1 GCD-v2 + 11 v7 + 4 v8`;
- all 100 baseline entries, including independent RDKit and class rescoring of
  all 60 SMILES entries;
- all 20 exact max+1 accuracy thresholds and the 90% syntax cap;
- the two frozen recovery histories, remaining-call accounting, and exact
  675-call total;
- the three held-out CSD paths, SHA-256 pins, and qualifying attempts;
- strict queue phases `10 / 2 / 2 / 3 / 3` and the exact GPU scope `0,2,3`;
- a green adjacent test run of 156 tests, an active synthesis block, and no
  synthesis process at review time.

To reuse this approval, pass the JSON companion to the corrected queue's
`--corrected-approval` option. The queue rechecks both artifact hashes before
startup and before every dispatch.
