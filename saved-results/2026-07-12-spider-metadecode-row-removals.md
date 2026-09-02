# Spider metaDecode row removals

Date: 2026-07-12 IST

The Spider Qwen3.5-4B and Qwen3.5-9B metaDecode rows were removed from
`results_matrix.md` after their approved re-synthesis cycle exhausted attempt
40 without producing an accepted strategy that beat the required baseline
goal. Per the user's standing rule, a row is discarded after this one failed
re-synthesis cycle rather than retained as a losing metaDecode result.

Removed rows:

- Qwen3.5-4B Spider metaDecode: prior matrix row reported 53.7% train accuracy;
  the re-synthesis exhausted at attempt 40 with no win (best reported 40.3%).
- Qwen3.5-9B Spider metaDecode: prior matrix row reported 64.7% held-out
  accuracy; the re-synthesis exhausted at attempt 40 with no win (best reported
  49.3%).

Baseline rows and unrelated GSM/SMILES metaDecode rows were not removed.
