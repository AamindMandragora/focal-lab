#!/usr/bin/env bash
# Install amazon-science/mxeval into the active conda env (or current Python).
# PyPI has no usable mxeval wheel; upstream omits data/ from wheels; we sync data/
# into site-packages and fix console_scripts for modern pip/setuptools.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/environment/vendor/mxeval"
REPO="https://github.com/amazon-science/mxeval.git"

if [[ ! -f "$VENDOR/setup.py" ]]; then
  mkdir -p "$(dirname "$VENDOR")"
  git clone --depth 1 "$REPO" "$VENDOR"
fi

if ! grep -q "evaluate_functional_correctness:main" "$VENDOR/setup.py"; then
  sed -i.bak 's/evaluate_functional_correctness = mxeval.evaluate_functional_correctness"/evaluate_functional_correctness = mxeval.evaluate_functional_correctness:main"/' "$VENDOR/setup.py"
fi

SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
rm -rf "${SITE:?}/data"
cp -a "$VENDOR/data" "$SITE/"
python -m pip install "$VENDOR" --no-build-isolation --no-deps
python -c "from mxeval.data import write_jsonl; print('mxeval import OK')"
