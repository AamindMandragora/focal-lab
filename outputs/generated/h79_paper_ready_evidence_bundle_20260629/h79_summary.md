# H79 paper-ready evidence bundle

Created UTC: 2026-06-29T22:53:54Z

This CPU-only bundle refreshes H74 after H78 added Spider-9B as a held-out win. It makes no model calls, GPU calls, billed API calls, or score-artifact edits.

Current packaged paper-ready wins: **3**

## Spider-2B

Status: paper_ready_heldout_win
Baseline bar: `{'accuracy': 0.377, 'syntax_rate': 0.907, 'source': 'IterGen held-out bar recorded in campaign docs'}`
Held-out metrics: `{'accuracy': None, 'syntax_rate': None}`
train_artifact: `outputs/generated/synth_spider_2b_seed334train300_0627c/synth_spider_2b_seed334train300_0627c_20260627_133305_7ab948/results/success_report.json` sha256 `7dbb589ccbe5c2c638837cb19571227732c4d5e08ffcae050417b89fdb12dde5` mtime `2026-06-27T18:44:04Z`
heldout_artifact: `outputs/generated/reeval_spider_2b_att17_HELDOUT_test300_0628/reeval_spider_2b_att17_HELDOUT_test300_0628_20260628_020452_5250f2/results/success_report.json` sha256 `4b91f1eb18e105a160e03f8d8b21378001ff5d2d52d69000685f1e7ef7a14911` mtime `2026-06-28T02:20:19Z`

## Spider-9B

Status: paper_ready_heldout_win
Baseline bar: `{'accuracy': 0.67, 'syntax_rate': 0.983, 'source': 'IterGen held-out bar recorded in campaign docs'}`
Held-out metrics: `{'accuracy': 0.74, 'syntax_rate': 0.99, 'num_examples': 300, 'total_output_tokens': 7443, 'max_sample_time_seconds': 36.9992}`
heldout_artifact: `outputs/generated/h78_spider9b_h19_aliasclean_heldout_20260629/h78_reeval.json` sha256 `2ed333bb03b7c4415b4d05aa35eb803cf44a86fcd925007c070f5cc1d45a57b8` mtime `2026-06-29T22:48:58Z`

## SMILES isocyanates-4B primary UV

Status: paper_ready_primary_uv_heldout_win
Baseline bar: `{'accuracy': 0.16, 'syntax_rate': 1.0, 'source': 'outputs/controlled_comparison/smiles_qwen35/4B/isocyanates/cars.json'}`
Held-out metrics: `{'accuracy': 0.58, 'syntax_rate': 0.61, 'num_examples': None}`
train_artifact: `outputs/generated/smiles_qwen35_4b_isocyanates_uv_qwen35_0627/smiles_qwen35_4b_isocyanates_uv_qwen35_0627_20260629_172330_b62714/results/success_report.json` sha256 `7d5ee9178e1aa3c95d1d409a7aa16a019ba5d0330969a754304b0a2de356df37` mtime `2026-06-29T18:51:25Z`
heldout_artifact: `outputs/controlled_comparison/smiles_qwen35_4b/isocyanates/metadecode_uv.json` sha256 `42257a2d5da58f44a0cfd845b43c25c1c1dd7f0492ddb5554ec724e659c3bbbf` mtime `2026-06-29T19:03:18Z`
cars_artifact: `outputs/controlled_comparison/smiles_qwen35/4B/isocyanates/cars.json` sha256 `47b28144eec14f0cfb0c997607acdddbf18a5ad10f659d798a7552b04beba55a` mtime `2026-06-26T09:50:23Z`
Caveat: Primary UV/accuracy clears live CARS UV bar; validity/syntax does not match the auxiliary CARS 1.00 value, so do not claim perfect validity.

