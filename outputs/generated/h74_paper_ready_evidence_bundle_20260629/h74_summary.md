# H74 Paper-ready evidence bundle

Created: 2026-06-29T21:01:40.320464+00:00
Refined: 2026-06-29T21:02:51.559544+00:00

No model, GPU, or billed API calls. No score artifacts edited.

Packaged wins: **2**

## Spider-2B

Status: `paper_ready_evidence_complete`

### Metrics

- `train_metrics`: `{"accuracy": 0.39, "num_correct": 117, "num_examples": 300, "success": true, "syntax_rate": 0.99, "total_attempts": 17}`
- `heldout_metrics`: `{"accuracy": 0.38333333333333336, "num_correct": 115, "num_examples": 300, "success": true, "syntax_rate": 0.9933333333333333, "total_attempts": 1}`
- `heldout_accuracy_count`: `115/300`

### Artifacts and sha256

- `outputs/generated/reeval_spider_2b_att17_HELDOUT_test300_0628/launch.log`: `5158418e7be14fec8261465628e46d9701987ee3ec4fec35e7cb91470bd83422`
- `outputs/generated/reeval_spider_2b_att17_HELDOUT_test300_0628/reeval_spider_2b_att17_HELDOUT_test300_0628_20260628_020452_5250f2/results/success_report.json`: `4b91f1eb18e105a160e03f8d8b21378001ff5d2d52d69000685f1e7ef7a14911`
- `outputs/generated/synth_spider_2b_seed334train300_0627c/synth_spider_2b_seed334train300_0627c_20260627_133305_7ab948/results/success_report.json`: `7dbb589ccbe5c2c638837cb19571227732c4d5e08ffcae050417b89fdb12dde5`

Caveat: Spider-2B exact comparison baseline is inherited from the existing success report/docs; bundle records current proof artifacts and hashes, not a new re-score.

## SMILES isocyanates-4B primary UV

Status: `paper_ready_primary_uv_evidence_complete`

### Metrics

- `train_metrics`: `{"accuracy": 0.48, "num_correct": 32, "num_examples": 50, "success": true, "syntax_rate": 0.64, "total_attempts": 12}`
- `heldout_metrics`: `{"accuracy": 0.58, "syntax_rate": 0.61}`
- `cars_metrics`: `{"accuracy": 0.16, "syntax_rate": 1.0}`
- `primary_uv_margin`: `0.41999999999999993`

### Artifacts and sha256

- `outputs/controlled_comparison/smiles_qwen35/4B/isocyanates/cars.json`: `47b28144eec14f0cfb0c997607acdddbf18a5ad10f659d798a7552b04beba55a`
- `outputs/controlled_comparison/smiles_qwen35_4b/isocyanates/metadecode_uv.json`: `42257a2d5da58f44a0cfd845b43c25c1c1dd7f0492ddb5554ec724e659c3bbbf`
- `outputs/generated/smiles_qwen35_4b_isocyanates_uv_qwen35_0627/smiles_qwen35_4b_isocyanates_uv_qwen35_0627_20260629_172330_b62714/results/success_report.json`: `7d5ee9178e1aa3c95d1d409a7aa16a019ba5d0330969a754304b0a2de356df37`

Caveat: Primary UV win only; validity/syntax is reported separately and does not match the CARS auxiliary validity value.

