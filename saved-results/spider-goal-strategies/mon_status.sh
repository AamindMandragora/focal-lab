#!/usr/bin/env bash
# Compact status of the two iter50 synthesis runs, one line each. Read by the local Monitor poll loop.
cd ~/csd-generation
for tag in spider7b spider1p5b; do
  L=$(ls -t logs/${tag}_iter50_cold_*.log 2>/dev/null | head -1)
  if [ -z "$L" ]; then echo "${tag}: no log yet"; continue; fi
  anchor=$(grep -aoE "anchor for next refinement: attempt [0-9]+ \(acc=[0-9.]+%?, syn(tax)?=[0-9.]+%?\)" "$L" | tail -1)
  if [ -z "$anchor" ]; then
    anchor=$(grep -aoE "attempt [0-9]+/[0-9]+|init engine|EngineCore|Started eval|evaluating|Loading checkpoint" "$L" | tail -1)
  fi
  err=$(grep -aE "Traceback|ThrottlingException|Throttling|NoCredentials|AccessDenied|CUDA out of memory|Killed|RuntimeError|MemoryError" "$L" | tail -1 | cut -c1-110)
  alive=$(pgrep -f "output-name ${tag}_iter50" >/dev/null && echo RUN || echo DEAD)
  # Only surface a [done] marker once the run is actually DEAD. While it is alive,
  # grepping all iter50_driver_*.log picks up a stale [done] from an earlier killed
  # run and falsely tags the live run as finished.
  done=""
  if [ "$alive" = DEAD ]; then
    done=$(grep -aE "\[done\] ${tag}" logs/iter50_driver_*.log 2>/dev/null | tail -1)
  fi
  echo "${tag} [${alive}] ${anchor:-(starting up)}${err:+ | ERR: $err}${done:+ | $done}"
done
