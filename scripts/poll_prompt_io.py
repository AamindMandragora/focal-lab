#!/usr/bin/env python3
"""Poll the most recent thinking_effort_high_logio_*/io/prompt_io.jsonl for new
Opus API records and emit one summary line per new record.

Emits to stdout (line-buffered, Monitor-friendly). Designed to run on focal."""
import glob
import json
import os
import sys
import time

FILE_GLOB = "/home/aadivyar/csd-generation/logs/thinking_effort_high_logio_*/prompt_io.jsonl"


def newest_io_file():
    files = sorted(glob.glob(FILE_GLOB), key=os.path.getmtime)
    return files[-1] if files else None


def summarize(rec, idx):
    out = rec.get("output", "") or ""
    usr = rec.get("user_prompt", "") or ""
    sysp = rec.get("system_prompt", "") or ""
    out_lines = out.count("\n") + 1 if out else 0
    method_blocks = out.count("method GenerateText")
    has_dafny = ("predicate " in out) or ("method " in out)
    has_delim_in_strategy = ("<<" in out and ">>" in out)
    # length proxies for "simpler vs over-engineered"
    helper_count = out.count("EnterObservedConstrainedSpan") + out.count("UnconstrainedChunk")
    branch_count = out.count(" if ") + out.count("\nif ")
    return (
        f"CALL#{idx} out_chars={len(out)} out_lines={out_lines} "
        f"method_blocks={method_blocks} has_dafny={has_dafny} "
        f"delim_in_code={has_delim_in_strategy} "
        f"helpers={helper_count} branches={branch_count} "
        f"usr_chars={len(usr)} sys_chars={len(sysp)}"
    )


def main():
    prev_count = 0
    waited_for_file = False
    while True:
        path = newest_io_file()
        if path is None:
            if not waited_for_file:
                print("waiting for prompt_io.jsonl to be created...", flush=True)
                waited_for_file = True
            time.sleep(15)
            continue
        if waited_for_file:
            print(f"found file: {path}", flush=True)
            waited_for_file = False
        try:
            with open(path) as f:
                lines = f.readlines()
        except FileNotFoundError:
            time.sleep(15)
            continue
        cur = len(lines)
        if cur > prev_count:
            for i in range(prev_count, cur):
                try:
                    rec = json.loads(lines[i])
                    print(summarize(rec, i + 1), flush=True)
                except Exception as e:
                    print(f"CALL#{i+1} PARSE_ERROR: {e}", flush=True)
            prev_count = cur
        time.sleep(20)


if __name__ == "__main__":
    main()
