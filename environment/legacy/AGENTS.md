# AGENTS.md — `environment/legacy/`

## Scope

Policies for **local upstream clones** under **`legacy/CRANE`** and **`legacy/itergen`** (gitignored trees used by **`run_legacy_fixed_strategy`**).

## Rule: edits must become tracked patches

If you **modify any file** under those **`legacy/*`** directories (upstream bugfixes, SynCode/Lark compatibility, pin tweaks, etc.), you **must**:

1. **Capture the delta** as one or more unified diff files under **`environment/legacy_patches/<CRANE|itergen>/`**, using **`git diff`** / **`git format-patch`** against a pristine upstream checkout at the same ref, or equivalent **`diff -u`** output with **`-p1`** applicability from that repo root.
2. **Name patches** so apply order is obvious (e.g. `010-lark-deepcopy.patch` before `020-…`).
3. **Verify** a clean reinstall: remove or rename the local **`legacy/<name>`** tree, run **`bash environment/clone_legacy_csds.sh`**, confirm patches apply cleanly, then optionally **`python synthesis/scripts/report_legacy_upstream_diff.py --fetch-upstream`** so only intentional differences remain.
4. **Document behavior** in **`environment/legacy/DIFFERENCES.md`** when the change affects baseline semantics or reproducibility (not for purely mechanical typo fixes if already obvious from the patch header).

Do **not** leave reproducibility-critical fixes **only** inside gitignored **`legacy/*`** with no patch in **`environment/legacy_patches/`**.

## Prefer harness changes when possible

If the same outcome can be achieved by changing **`synthesis/evaluate/run_legacy_fixed_strategy.py`** (or shared grammars / evaluator wiring) without forking upstream, prefer that so fewer patches need maintenance.

## See also

- **`DIFFERENCES.md`** — harness vs upstream behavior (orthogonal to file-level patches).
- **`repos.json`** — default upstream URLs for clones and diff tooling.
- **`../legacy_patches/README.md`** — patch format and apply semantics.
