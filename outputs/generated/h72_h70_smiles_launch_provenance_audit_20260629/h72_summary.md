# H72 H70 SMILES launch provenance audit

Checked: 2026-06-29T20:51:39.260989+00:00

Conclusion: `provenance_risk_needs_isolation_or_prelaunch_snapshot`

## Key facts

- `planned_generated_root_matches_old_root`: `True`
- `planned_heldout_matches_current_heldout`: `True`
- `overwrites_or_updates_current_heldout_path`: `True`
- `uses_old_generated_root`: `True`
- `pilot_mentions_latest_run`: `True`
- `pilot_mentions_controlled_comparison`: `True`

## Recommended mitigation
- Before real H70 launch, snapshot current held-out JSON and latest_run metadata under the H70 artifact directory.
- Prefer patching/materializing H70 to set or copy a unique H70 output root if pilot script supports it; otherwise record the prelaunch latest_run and heldout hash, then after launch record the new timestamped run dir and overwrite delta.
- Do not claim H70 result unless the post-launch held-out JSON and train success report are copied into an H70-specific artifact folder.
