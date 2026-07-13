#!/usr/bin/env bash
# Install amazon-science/mxeval into the active conda env (or current Python).
# PyPI has no usable mxeval wheel; upstream omits data/ from wheels; we sync data/
# into site-packages and fix console_scripts for modern pip/setuptools.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/environment/vendor/mxeval"
REPO="https://github.com/amazon-science/mxeval.git"
MXEVAL_COMMIT="${MXEVAL_COMMIT:-e09974f990eeaf0c0e8f2b5eaff4be66effb2c86}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d "$VENDOR/.git" ]]; then
  mkdir -p "$(dirname "$VENDOR")"
  git init "$VENDOR"
  git -C "$VENDOR" remote add origin "$REPO"
fi
git -C "$VENDOR" fetch --depth 1 origin "$MXEVAL_COMMIT"
if [[ "$(git -C "$VENDOR" rev-parse HEAD 2>/dev/null || true)" != "$MXEVAL_COMMIT" ]]; then
  git -C "$VENDOR" checkout --detach --force FETCH_HEAD
fi
if [[ "$(git -C "$VENDOR" rev-parse HEAD)" != "$MXEVAL_COMMIT" ]]; then
  echo "mxeval checkout does not match pinned commit: $MXEVAL_COMMIT" >&2
  exit 1
fi

if ! grep -q "evaluate_functional_correctness:main" "$VENDOR/setup.py"; then
  sed -i.bak 's/evaluate_functional_correctness = mxeval.evaluate_functional_correctness"/evaluate_functional_correctness = mxeval.evaluate_functional_correctness:main"/' "$VENDOR/setup.py"
fi

SITE="$("$PYTHON_BIN" -c 'import site; print(site.getsitepackages()[0])')"
rm -rf "${SITE:?}/data"
cp -a "$VENDOR/data" "$SITE/"
"$PYTHON_BIN" -m pip install "$VENDOR" --no-build-isolation --no-deps
"$PYTHON_BIN" -c "from mxeval.data import write_jsonl; print('mxeval import OK')"
