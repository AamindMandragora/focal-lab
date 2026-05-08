# Baselines Artifacts

This directory stores baseline evaluation outputs used for comparison against synthesized strategies.

## Intended Contents

Store one JSON file per baseline strategy, model, and benchmark pair.

Suggested structure:

- `outputs/baselines/<strategy_name>/<model_name>/<benchmark_name>.json`

Each JSON should contain only:

- `accuracy`
- `syntax_rate`
- `answers` (every generated answer keyed or listed by benchmark question)

You can generate this file from a synthesis success report with:

```bash
python -m synthesis.evaluate.export_baseline_json \
  --success-report <path/to/success_report.json> \
  --output outputs/baselines/<strategy>/<model>/<benchmark>.json
```
