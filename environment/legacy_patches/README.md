# Optional patches applied after cloning legacy baselines

After **`environment/clone_legacy_csds.sh`** clones into **`legacy/`**, it applies
any **`*.patch`** files found here, per upstream tree:

- `environment/legacy_patches/CRANE/*.patch`
- `environment/legacy_patches/itergen/*.patch`
- `environment/legacy_patches/cars/*.patch`
- `itergen/010-sign-aware-recurrence-penalty.patch` fixes negative-logit
  recurrence handling and carries its regression test.
- `itergen/011-empty-config-cache-full-prompt.patch` keeps the complete prompt
  until a config-allocated Qwen3.5 cache contains an actual token.

Patches should be **`git format-patch`**-style or **`git diff`** unified diffs
generated **relative to the patched repository root** (prefix level `-p1`).

If `git apply` fails (e.g. non-git snapshot), the clone script falls back to
**`patch -p1`**.

Keep patches **minimal** and document motivation in the patch header or in
**`environment/legacy/DIFFERENCES.md`**.

Agent/human policy for edits in **`legacy/`** trees: **`../legacy/AGENTS.md`**.
