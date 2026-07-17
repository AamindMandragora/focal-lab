# Spider hourly warm restart

Date: 2026-07-12 IST

The user explicitly approved a one-time warm-restart exception for three Spider
cells and approved charging UIUC lab AWS account `887730490125` in `us-east-1`.
The credential source is focal `/home/aadivyar/csd-generation/.env` and the
secret value was not printed or copied into this record.

Restart points:

- Spider Qwen3.5-4B: replay completed attempt 39, then continue through 40.
- Spider Qwen3.5-9B: replay completed attempt 37, then continue through 40.
- Spider Qwen2.5-7B: replay completed attempt 7, then continue through 40.

All three retain their original model, seed334 train-300 split, acceptance bars,
evaluation limits, adaptive helper mask, bandit policy, beam size 2, and output
names. Only the retry policy changes to indefinite retries centered on one hour.

The same explicit continuation approval was later extended to GSM Qwen3.5-9B
attempt 5 and GSM Qwen2.5-14B attempt 33. The deferred controller is
`/home/aadivyar/csd-generation/.context/run_deferred_gsm_resume_controller.sh`.
It replays both strategies to reconstruct evaluation failure feedback before
new author calls.

The controller also restores prior evaluated-attempt history before replay:
GSM 9B restores attempts 1-4 (including anchor attempt 3), and GSM 14B restores
30 evaluated attempts from the 1-32 range (including anchor attempt 12). This
preserves the earlier best-strategy and helper-selection context in the next
refinement prompt rather than retaining only the latest attempt.
