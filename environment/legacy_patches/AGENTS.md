# AGENTS.md — `environment/legacy_patches/`

## Scope

**Tracked unified diffs** applied to gitignored **`legacy/`** clones after **`clone_legacy_csds.sh`**.

## Rules

- Any reproducibility-critical edit under **`legacy/{CRANE,itergen,cars}`** must have a matching patch here.
- Use **`git diff`** / **`git format-patch`** with **`-p1`** applicability from the upstream repo root.
- Name patches for apply order (e.g. `010-…`, `020-…`).
- Document behavioral impact in **`../legacy/DIFFERENCES.md`** when not obvious from the patch header.

## See also

- **`README.md`** in this folder.
- **`../legacy/AGENTS.md`**.
