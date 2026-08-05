# Corrected full-baseline queue launch

Date: 2026-08-05
Launch time: 12:54:20 UTC

## Outcome

The independently approved corrected synthesis queue is running on focal.

- Controller PID: `3111473`
- Controller process group: `3111473`
- Queue profile: `full-baseline-corrected-20260805`
- Allowed physical GPUs: `0,2,3`
- GPU `1` compute processes at verification: `0`
- Planned new author calls: `675`
- Controller log: `logs/full-baseline-corrected-20260805-cold-controller.log`
- Combined run log: `logs/full-baseline-corrected-20260805-combined.log`
- State directory: `.context/full-baseline-corrected-20260805-cold-state`
- PID file: `.context/full-baseline-corrected-20260805-cold.pid`

## Bound inputs

- Pinned code commit: `6766ebd397f4bcdbd4cc3332e1051a9efd6258a6`
- Corrected evidence SHA-256:
  `57392e149cea23efe6f596b921c6cd3d74ae7519e7c286393adadc9bdb579ab7`
- Queue manifest SHA-256:
  `c5f5683311ecf8f03211fb4fd8572b92e472dc5475367afe72413eb31098aa18`
- Approval JSON SHA-256:
  `d43a5cef6c9484430fc87e2a7dca4e6bdf963148d0ff5a7769c6422b4a602b9a`
- Archived block SHA-256:
  `c882e6ae8178349ea6006933d17b1b8105c73a20612c8d800ec79df16057a4a1`

## Launch transition

1. The old exact-zero monitor PID `336923` was checked as its own process-group
   leader and stopped with `SIGTERM`; it was confirmed dead before launch.
2. The active block file was moved without changing its bytes to
   `saved-results/2026-08-05-exact-zero-repair-synthesis-block-resolved.json`.
3. The controller started with the exact approval JSON, manifest, code commit,
   and `--gpus 0,2,3` gate.
4. The controller and three phase-one synthesis children remained alive through
   the post-launch check. Their physical GPU environments were `3,0`, `2`, and
   `3`; none included GPU `1`.
5. Approval revalidation returned the exact `675`-call plan after launch.

## First phase observed

- `spider-qwen35-2b`: GPUs `3,0`, 40 attempts.
- `smiles-acrylates-qwen35-2b`: GPU `2`, 40 attempts.
- `smiles-chain_extenders-qwen25-1p5b`: GPU `3`, 40 attempts.

The controller may share an approved GPU when its memory reservations fit; the
queue never considers GPU `1`. Later phases remain blocked by the strict phase
barrier until every earlier-phase job has reached a terminal state.
