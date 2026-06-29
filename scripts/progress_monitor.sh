#!/usr/bin/env bash
# Auto-updating progress monitor for the experimental matrix.
# Writes a markdown summary to ~/csd-generation/progress.md every 5 minutes.
set -uo pipefail
PROJ="$HOME/csd-generation"
OUT="$PROJ/progress.md"
LOGDIR="$PROJ/logs/tmux"
PY=/apps/conda/advayth2/envs/advayth2/bin/python
LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib

while true; do
  LATEST_LOG=$(ls -t "$LOGDIR"/full_matrix_*.log 2>/dev/null | head -1)
  LATEST_LAUNCHER=$(ls -t "$PROJ"/scripts/launch_full_matrix_*.sh 2>/dev/null | head -1)

  {
    echo "# Experiment progress"
    echo
    echo "_Last updated: $(date -Is)_"
    echo

    # Run state
    if pgrep -u aadivyar -f run_all_tests.py >/dev/null 2>&1; then
      RUN_PID=$(pgrep -u aadivyar -f run_all_tests.py | head -1)
      RUN_ETIME=$(ps -o etime= -p "$RUN_PID" | tr -d ' ')
      echo "## Status: ✅ RUN ACTIVE  (PID $RUN_PID, uptime $RUN_ETIME)"
    elif [[ -n "$LATEST_LAUNCHER" && -n "$LATEST_LOG" ]]; then
      echo "## Status: ⏹  NO RUN ACTIVE  (last launcher: $(basename "$LATEST_LAUNCHER"))"
    else
      echo "## Status: (no run yet)"
    fi
    echo

    # GPU
    echo "## GPU"
    echo
    echo '```'
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null
    echo '```'
    echo

    # Completed cells
    if [[ -n "$LATEST_LAUNCHER" ]]; then
      echo "## Completed cells (since $(basename "$LATEST_LAUNCHER"))"
      echo
      echo "| Strategy | Model | Benchmark | Acc | Syntax | N | Saved |"
      echo "|---|---|---|---|---|---|---|"
      find "$PROJ/outputs/baselines" -name "*.json" -newer "$LATEST_LAUNCHER" 2>/dev/null \
        | sort \
        | while read -r f; do
            $PY - <<PY 2>/dev/null
import json, os, sys
f = "$f"
try:
    d = json.load(open(f))
except Exception as e:
    print(f"| ? | ? | ? | err | err | ? | err |")
    sys.exit(0)
rel = f.replace("$PROJ/outputs/baselines/","")
strategy, model, fname = rel.split("/", 2)
bench = fname.split("__")[0]
acc = d.get("accuracy")
syn = d.get("metrics",{}).get("syntax_rate")
n   = d.get("metrics",{}).get("sample_size") or len(d.get("answers", []))
acc_s = f"{acc*100:.1f}%" if isinstance(acc,(int,float)) else "?"
syn_s = f"{syn*100:.1f}%" if isinstance(syn,(int,float)) else "?"
mtime = os.path.getmtime(f)
import datetime
ts = datetime.datetime.fromtimestamp(mtime).isoformat(timespec='seconds')
print(f"| {strategy} | {model} | {bench} | {acc_s} | {syn_s} | {n} | {ts} |")
PY
          done
      echo
    fi

    # Errors found in log
    if [[ -n "$LATEST_LOG" ]]; then
      echo "## Errors in latest log"
      echo
      ERR_COUNT=$(grep -cE "Traceback|ModuleNotFoundError|CalledProcessError|RuntimeError|TypeError|CXXABI" "$LATEST_LOG" 2>/dev/null || echo 0)
      echo "Total error/traceback lines: **$ERR_COUNT**"
      echo
      if [[ "$ERR_COUNT" -gt 0 ]]; then
        echo "Unique error types:"
        echo '```'
        grep -E "Error:|RuntimeError|AssertionError|ValueError|TypeError|KeyError|IndexError|NotImplementedError|AttributeError|ModuleNotFoundError|CXXABI" "$LATEST_LOG" 2>/dev/null | sort -u | head -15
        echo '```'
      fi
      echo

      echo "## Latest log tail"
      echo
      echo '```'
      tail -25 "$LATEST_LOG"
      echo '```'
    fi
  } > "$OUT.tmp"
  mv "$OUT.tmp" "$OUT"

  sleep 300
done
