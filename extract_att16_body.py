#!/usr/bin/env python3
"""
Extract attempt-16 strategy body from fallback_winner.json and write it in
--initial-strategy-file format.

Body format rules (from project CLAUDE.md):
  - Must START with // CSD_RATIONALE_BEGIN (with leading 4 spaces, matching reference)
  - No class wrapper (strip header above the rationale comment)
  - No `var helpers` line
  - Strip the last 2 closing braces that close the class + module

Run from /home/aadivyar/csd-generation:
  python3 extract_att16_body.py
"""

import json
import glob
import os
import sys

BASE = "/home/aadivyar/csd-generation"
OUTPUT_BASE = os.path.join(BASE, "outputs/generated/gsm1p5b_seed123_fresh_20260611")
OUT_BODY = os.path.join(BASE, "gsm1p5b_seed123_att16_body.dfy")

# ── Find newest timestamp directory ──────────────────────────────────────────
timestamp_dirs = sorted(glob.glob(os.path.join(OUTPUT_BASE, "*")))
if not timestamp_dirs:
    print(f"ERROR: no timestamp dirs found under {OUTPUT_BASE}", file=sys.stderr)
    sys.exit(1)
newest = timestamp_dirs[-1]
print(f"Using timestamp dir: {newest}")

winner_path = os.path.join(newest, "results", "fallback_winner.json")
if not os.path.exists(winner_path):
    print(f"ERROR: fallback_winner.json not found at {winner_path}", file=sys.stderr)
    sys.exit(1)

print(f"Loading {winner_path} ...")

# Stream-parse: file is 16MB — load it but only touch what we need
with open(winner_path, "r") as f:
    data = json.load(f)

# ── Locate attempt 16 ────────────────────────────────────────────────────────
# Structure: data may be a list of attempts, or a dict with an "attempts" key,
# or a dict with "strategy_code" directly if it already IS the winner attempt.
def find_attempt_16(data):
    """Return the strategy_code string for attempt index 16 (0- or 1-based)."""
    # Case 1: top-level has strategy_code directly (it IS the winner)
    if isinstance(data, dict) and "strategy_code" in data:
        attempt_num = data.get("attempt_number") or data.get("attempt_index")
        print(f"Top-level winner: attempt_number={attempt_num}")
        return data["strategy_code"], attempt_num

    # Case 2: list of attempt dicts
    if isinstance(data, list):
        attempts = data
    elif isinstance(data, dict):
        # Try common keys
        for key in ("attempts", "history", "results", "all_attempts"):
            if key in data and isinstance(data[key], list):
                attempts = data[key]
                break
        else:
            # Dump top-level keys for debugging
            print(f"Top-level keys: {list(data.keys())[:20]}", file=sys.stderr)
            sys.exit(1)

    print(f"Total attempts in file: {len(attempts)}")

    # Find attempt 16 — check both 0-based index 15 and 1-based attempt_number 16
    for i, att in enumerate(attempts):
        num = att.get("attempt_number") or att.get("attempt_index") or att.get("iteration")
        if num == 16 or i == 15:
            print(f"Found at list index {i}, attempt_number={num}")
            code = att.get("strategy_code") or att.get("code") or att.get("dafny_code")
            if code is None:
                print(f"Keys in attempt: {list(att.keys())[:20]}", file=sys.stderr)
                sys.exit(1)
            return code, num

    print(f"ERROR: attempt 16 not found (tried index 15 and attempt_number==16)", file=sys.stderr)
    print(f"Available attempt numbers: {[a.get('attempt_number') or a.get('attempt_index') or a.get('iteration') for a in attempts[:20]]}", file=sys.stderr)
    sys.exit(1)


raw_code, found_num = find_attempt_16(data)
print(f"Strategy code length: {len(raw_code)} chars")

# ── Convert to body format ────────────────────────────────────────────────────
# Full .dfy format looks like:
#
#   module GeneratedStrategy {
#     class CSDStrategy {
#       method GenerateCSD(...) {
#         var helpers := ...  ← strip this line
#         // CSD_RATIONALE_BEGIN
#         ...body...
#         }   ← strip (closes method)
#       }     ← strip (closes class)
#   }         ← strip (closes module) BUT only last 2 braces per rule
#
# Rule: strip everything ABOVE "// CSD_RATIONALE_BEGIN",
#       strip the `var helpers` line if present,
#       strip the last 2 closing-brace lines.

lines = raw_code.splitlines()

# Find the rationale start line
rationale_start = None
for i, line in enumerate(lines):
    if "CSD_RATIONALE_BEGIN" in line:
        rationale_start = i
        break

if rationale_start is None:
    print("ERROR: CSD_RATIONALE_BEGIN not found in strategy_code", file=sys.stderr)
    print("First 10 lines:", file=sys.stderr)
    for l in lines[:10]:
        print(f"  {repr(l)}", file=sys.stderr)
    sys.exit(1)

body_lines = lines[rationale_start:]

# Strip `var helpers` line if present (must not appear in body format)
body_lines = [l for l in body_lines if not l.strip().startswith("var helpers")]

# Strip last 2 closing-brace lines (close class + close module)
# Walk from the end, skip blank lines, remove 2 lines that are only `}`
stripped = 0
result_lines = list(body_lines)
i = len(result_lines) - 1
while i >= 0 and stripped < 2:
    stripped_line = result_lines[i].strip()
    if stripped_line == "}":
        result_lines.pop(i)
        stripped += 1
    elif stripped_line == "":
        pass  # skip blank, keep scanning
    else:
        break  # hit real content, stop
    i -= 1

if stripped < 2:
    print(f"WARNING: only stripped {stripped} closing braces (expected 2). Check output manually.", file=sys.stderr)

body = "\n".join(result_lines)

# ── Sanity checks ─────────────────────────────────────────────────────────────
first_line = result_lines[0] if result_lines else ""
assert "CSD_RATIONALE_BEGIN" in first_line, f"First line should be CSD_RATIONALE_BEGIN, got: {repr(first_line)}"

has_class = any("class " in l for l in result_lines)
if has_class:
    print("WARNING: 'class ' still present in body — check stripping", file=sys.stderr)

has_var_helpers = any(l.strip().startswith("var helpers") for l in result_lines)
if has_var_helpers:
    print("WARNING: 'var helpers' still present in body — check stripping", file=sys.stderr)

print(f"Body line count: {len(result_lines)}")
print(f"First line: {repr(first_line)}")
print(f"Last line:  {repr(result_lines[-1]) if result_lines else '(empty)'}")
print(f"'class ' present: {has_class}")
print(f"'var helpers' present: {has_var_helpers}")
print(f"Closing braces stripped: {stripped}")

# ── Write output ──────────────────────────────────────────────────────────────
with open(OUT_BODY, "w") as f:
    f.write(body)
    if not body.endswith("\n"):
        f.write("\n")

print(f"\nWrote body to: {OUT_BODY}")
print("DONE")
