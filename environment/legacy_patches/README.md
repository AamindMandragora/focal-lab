# Optional patches applied after cloning legacy baselines

After **`environment/clone_legacy_csds.sh`** clones into **`legacy/`**, it applies
any **`*.patch`** files found here, per upstream tree:

- `environment/legacy_patches/CRANE/*.patch`
- `environment/legacy_patches/itergen/*.patch`
- `environment/legacy_patches/cars/*.patch`

Patches should be **`git format-patch`**-style or **`git diff`** unified diffs
generated **relative to the patched repository root** (prefix level `-p1`).

If `git apply` fails (e.g. non-git snapshot), the clone script falls back to
**`patch -p1`**.

Keep patches **minimal** and document motivation in the patch header or in
**`environment/legacy/DIFFERENCES.md`**.

Current apply order:

| Repo | Patches |
|------|---------|
| CRANE | `010-vas-prompt-tiers-base`, `020-vas-harness-utils`, `030-vas-smiles-prompt-state-grammar`, `040-vas-gsm-z3-only-equivalence` |
| itergen | `010-vas-cache-and-hf-env`, `020-vas-language-model-compat`, `030-vas-lark-tree-deepcopy`, `040-vas-main-harness`, `050-vas-readme-notes` |
| cars | `010-vas-lib-hf-cache-device-map`, `020-vas-mcmc-lib-hf-cache`, `030-vas-run-task-harness`, `040-vas-readme-notes` |

Agent/human policy for edits in **`legacy/`** trees: **`../legacy/AGENTS.md`**.
