import json, glob, os

os.chdir(os.path.expanduser("~/csd-generation"))

print("=== SUCCESS REPORTS ===")
for f in sorted(glob.glob("outputs/generated/*/results/success_report.json")):
    with open(f) as fh:
        d = json.load(fh)
    ev = d.get("evaluation_result", d.get("evaluation", {}))
    acc = ev.get("accuracy", 0)
    syn = ev.get("syntax_rate", 0)
    n = ev.get("num_examples", "?")
    att = d.get("total_attempts", d.get("attempts_needed", "?"))
    dirname = f.split("/")[2]
    print("SUCCESS | acc={:.2%} | syn={:.2%} | N={} | att={} | {}".format(acc, syn, n, att, dirname))

print("\n=== 7B FAILURE REPORTS ===")
for f in sorted(glob.glob("outputs/generated/*/results/failure_report.json")):
    with open(f) as fh:
        d = json.load(fh)
    best = d.get("best_attempt", {})
    ev = best.get("evaluation", {})
    if not ev:
        attempts = d.get("attempts", [])
        evals = [a.get("evaluation", {}) for a in attempts if a.get("evaluation")]
        if evals:
            ev = max(evals, key=lambda e: e.get("accuracy", 0))
    acc = ev.get("accuracy", 0)
    syn = ev.get("syntax_rate", 0)
    n = ev.get("num_examples", "?")
    dirname = f.split("/")[2]
    if "7B" in dirname:
        print("FAIL | best_acc={:.2%} | best_syn={:.2%} | N={} | {}".format(acc, syn, n, dirname))

print("\n=== ALL 1.5B FAILURE REPORTS ===")
for f in sorted(glob.glob("outputs/generated/*/results/failure_report.json")):
    with open(f) as fh:
        d = json.load(fh)
    best = d.get("best_attempt", {})
    ev = best.get("evaluation", {})
    if not ev:
        attempts = d.get("attempts", [])
        evals = [a.get("evaluation", {}) for a in attempts if a.get("evaluation")]
        if evals:
            ev = max(evals, key=lambda e: e.get("accuracy", 0))
    acc = ev.get("accuracy", 0)
    syn = ev.get("syntax_rate", 0)
    n = ev.get("num_examples", "?")
    dirname = f.split("/")[2]
    if "1.5B" in dirname:
        print("FAIL | best_acc={:.2%} | best_syn={:.2%} | N={} | {}".format(acc, syn, n, dirname))

print("\n=== BASELINE RESULTS ===")
for f in sorted(glob.glob("outputs/baselines/**/*.json", recursive=True)):
    try:
        with open(f) as fh:
            d = json.load(fh)
        acc = d.get("accuracy", d.get("acc", "?"))
        syn = d.get("syntax_rate", d.get("syntax", "?"))
        n = d.get("num_examples", d.get("n", d.get("total_examples", "?")))
        name = f.replace("outputs/baselines/", "")
        if isinstance(acc, float):
            print("BASELINE | acc={:.2%} | syn={} | N={} | {}".format(acc, syn, n, name))
        else:
            print("BASELINE | acc={} | syn={} | N={} | {}".format(acc, syn, n, name))
    except Exception as e:
        print("ERROR reading {}: {}".format(f, e))
