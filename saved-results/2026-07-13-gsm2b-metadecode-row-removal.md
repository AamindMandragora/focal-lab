# GSM Qwen3.5-2B metaDecode row removal

Date: 2026-07-13 IST

The GSM Qwen3.5-2B metaDecode result and its cancelled ablation provenance row
were removed from `results_matrix.md`. The approved one-cycle re-synthesis ran
through attempt 40 without producing a winning CSD. Its best re-synthesis
accuracy was 10.2% with 42.9% syntax; its best-syntax attempt had 2.0% accuracy
and 100% syntax. No held-out evaluation was warranted.

The Spider Qwen3.5-2B held-out metaDecode win remains in the matrix. GSM
baseline rows also remain.
