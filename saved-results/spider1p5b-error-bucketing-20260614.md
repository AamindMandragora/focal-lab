# Spider-1.5B Error Bucketing Analysis

**Date:** 2026-06-14
**Run analyzed:** `spider1p5b_300x300_seed334_heldout_dbfix_20260606`
**Result file:** `.../results/success_report.json` (300 held-out examples)
**Produced by:** throwaway Python script run in /tmp on focal (read-only analysis)

## Summary

- Total examples: 300
- Correct: 153 (51.0%)
- Wrong: 147 (49.0%)
- IterGen bar: 157/300 (52.3%) — need 4 more to beat it

## Constrained-path coverage

- `contains_delimiters=True`: 297/300 (99.0%) — constrained path firing almost universally
- `num_valid_visible_spans > 0`: 294/300 (98.0%)

## Error buckets (of 147 wrong examples)

| Bucket | N | % of wrong | % of 300 | Notes |
|---|---|---|---|---|
| syntax_invalid | 14 | 9.5% | 4.7% | 5 flagged `is_syntax_valid=False` (garbled output: `YOUR! QUERY`, truncated with `>》`); 9 exec errors other than schema (ambiguous column name, misuse of aggregate COUNT() in WHERE) |
| out_of_schema | 64 | 43.5% | 21.3% | All crash at sqlite3 execution with "no such table" or "no such column"; model hallucinates table names like `pets_1`, `concert_singer`, or references columns on the wrong table |
| semantic | 69 | 46.9% | 23.0% | SQL executes without error, all identifiers valid, but rows differ from gold; includes 21 alias-only false positives (SQL uses `AS alias_name` tokens that aren't schema refs — originally misclassified as out-of-schema) |
| unclassifiable | 0 | 0.0% | 0.0% | — |

**Check:** 14 + 64 + 69 + 0 = 147 ✓

## Headroom analysis

- **Constraint-addressable** (syntax_invalid + out_of_schema): **78 examples**
- If all 78 fixed: 153 + 78 = 231/300 = 77.0%
- **Need only 4 more** to clear IterGen's 157/300 bar
- 78 >> 4: large headroom exists in the constraint-addressable bucket

## Dominant failure pattern in out_of_schema (64 cases)

The model consistently hallucinates table names that don't exist in the DB (e.g. `pets_1` as a table name when the actual table is `pets`; `concert_singer` when the join table is `singer_in_concert`). It also references columns on wrong tables (e.g. `stadium.id` when the PK is `stadium_id`, `concert_id` on the `singer` table). These are JOIN construction errors — the model knows the right data is in the DB but misnames the join path.

## Methodology

- `db_id` was absent from the report JSON; recovered by joining on normalized question text to dev.json (1034 entries, 147/147 recovered)
- DB directory: `/home/aadivyar/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases/`
- dev.json: same syncode path, `evaluation_examples/examples/dev.json`
- Bucketing order: (1) `is_syntax_valid=False` flag → syntax_invalid; (2) sqlite3 execution error → schema or other; (3) post-exec identifier check (excluding AS aliases and gold SQL identifiers) → schema or semantic
- 21 examples originally flagged as out-of-schema (post-exec unknown ids) were reclassified as semantic after confirming all unknown tokens were AS alias names, not schema references
