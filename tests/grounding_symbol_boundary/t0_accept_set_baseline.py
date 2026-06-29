"""T0 byte-identical gate for the SymbolPosMap port (Task #24).

The SymbolPosMap port adds a pure side-record to the vendored syncode
IncrementalParser. It MUST NOT change which tokens are accepted (else other
benchmarks' decode shifts -> unfair). This harness feeds growing SQL and GSM
prefixes through a fresh IncrementalParser and records the accept-terminal sets
returned at each step. Run BEFORE the port (--out t0_before.json) and AFTER
(--out t0_after.json); the two JSONs must be IDENTICAL.

Model-free: exercises only the parser's accept-set computation (no vLLM/HF).

Usage (on focal):
  export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}
  export PYTHONPATH=/home/aadivyar/csd-generation
  cd /home/aadivyar/csd-generation
  python tests/grounding_symbol_boundary/t0_accept_set_baseline.py --out /tmp/t0_before.json
  # (apply port, scp file)
  python tests/grounding_symbol_boundary/t0_accept_set_baseline.py --out /tmp/t0_after.json
  python tests/grounding_symbol_boundary/t0_accept_set_baseline.py --compare /tmp/t0_before.json /tmp/t0_after.json
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path("/home/aadivyar/csd-generation")
GRAMMARS = REPO / "synthesis/evaluate/grammars"

# Put the vendored syncode on sys.path, exactly like parser_utils._ensure_syncode_import_path.
SYNCODE_DIR = REPO / "synthesis/evaluate/syncode"
if str(SYNCODE_DIR) in sys.path:
    sys.path.remove(str(SYNCODE_DIR))
sys.path.insert(0, str(SYNCODE_DIR))

# Growing prefixes (each list is fed cumulatively to ONE IncrementalParser, like
# the runtime does). Chosen to exercise NAMEs, JOINs, dotted refs, aliases,
# arithmetic, and << >> spans — the paths the SymbolPosMap updaters touch.
SQL_PREFIXES = [
    "SELECT",
    "SELECT name",
    "SELECT name FROM",
    "SELECT name FROM singer",
    "SELECT name FROM singer WHERE",
    "SELECT name FROM singer WHERE age",
    "SELECT name FROM singer WHERE age >",
    "SELECT name FROM singer WHERE age > 20",
    "SELECT T1.name FROM singer AS T1 JOIN concert AS T2 ON T1.id = T2.singer_id",
    "SELECT count(*) FROM stadium WHERE capacity > 5000 GROUP BY name",
]

GSM_PREFIXES = [
    "<<",
    "<<3",
    "<<3+",
    "<<3+4",
    "<<3+4>>",
    "<<3+4>> and then <<5*2>>",
    "The answer is <<10/2>>",
]


def _record_for_grammar(grammar_path, start, prefixes):
    from syncode.parsers.grammars import Grammar
    from syncode.parsers import create_base_parser
    from syncode.parsers.incremental_parser import IncrementalParser

    text = grammar_path.read_text()
    grammar = Grammar(text)
    base_parser = create_base_parser(grammar)
    ip = IncrementalParser(base_parser, ignore_whitespace=False)

    steps = []
    for code in prefixes:
        rec = {"code": code}
        try:
            ip.get_acceptable_next_terminals(code)
            rec["cur"] = sorted(str(t) for t in ip.cur_ac_terminals)
            rec["next"] = sorted(str(t) for t in ip.next_ac_terminals)
            rec["ok"] = True
        except Exception as e:  # record the exception TYPE so a behavior change shows
            rec["ok"] = False
            rec["err"] = type(e).__name__
        steps.append(rec)
    return steps


def build():
    out = {}
    out["sql"] = _record_for_grammar(GRAMMARS / "sql.lark", "start", SQL_PREFIXES)
    out["gsm"] = _record_for_grammar(GRAMMARS / "gsm.lark", "start", GSM_PREFIXES)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = ap.parse_args()

    if args.compare:
        a = json.loads(Path(args.compare[0]).read_text())
        b = json.loads(Path(args.compare[1]).read_text())
        if a == b:
            print("[PASS] T0: accept-sets BYTE-IDENTICAL before/after the port.")
            raise SystemExit(0)
        print("[FAIL] T0: accept-sets DIVERGED. Diffs:")
        for g in sorted(set(a) | set(b)):
            for i, (ra, rb) in enumerate(zip(a.get(g, []), b.get(g, []))):
                if ra != rb:
                    print(f"  [{g} step {i}] code={ra.get('code')!r}")
                    print(f"    before: {ra}")
                    print(f"    after:  {rb}")
        raise SystemExit(1)

    data = build()
    if args.out:
        Path(args.out).write_text(json.dumps(data, indent=2, sort_keys=True))
        print(f"wrote {args.out}")
    else:
        print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
