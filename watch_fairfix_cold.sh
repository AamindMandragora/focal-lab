#!/bin/bash
# Monitor filter for the Spider-1.5B fairfix COLD re-run (2026-06-17).
# Tails run.log and emits only summary/decision/failure lines so each becomes
# one chat event. Covers progress AND every terminal/failure signature
# (silence != success). tail -F handles the log being (re)created after launch.
LOG=/home/aadivyar/csd-generation/outputs/generated/spider1p5b_fairfix_cold_20260617/run.log
tail -F "$LOG" 2>/dev/null | stdbuf -oL grep -E \
  "Attempt [0-9]+/20|acc=[0-9]|Evaluation below threshold|Strategy accepted|ACCEPT|REJECT|new best|WIN|Span-closure check|grounded|Traceback|RuntimeError|CUDA error|OutOfMemory|OOM|Killed|FAILED|Exception|Error:|SYNTH_EXIT|DONE_SPIDER1P5B"
