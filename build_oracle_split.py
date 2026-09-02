"""Disposable: build a ~10-example probe split of Spider-7B STRUCTURAL failures
(in-schema, valid-but-wrong SQL) and record the reference gold + argmax answer
for each. Used by the self-consistency diversity probe. Read-only except for the
two small JSON files it writes. Runs on focal.

Inputs:
  - the 57.3% held-out success_report.json (argmax `actual` per example)
  - the seed334 full split (test_indices -> original Spider dev indices)
Outputs (written to /home/aadivyar/csd-generation/):
  - environment/benchmark_splits/spider_probe_struct_seed334.json  (full split
    dict, test_indices trimmed to the chosen structural-failure dev indices)
  - /tmp/sc_probe_reference.json  (per chosen example: question, gold, argmax SQL)
Algorithm: re-bucket the in-schema failures exactly like diagnostic #2, take the
SELECT_SAME bucket first (diff is purely WHERE/JOIN -> structural), top up with
SELECT_DIFF, map positions to test_indices, write the trimmed split + reference.
"""
import copy
import json
import re
import sys

sys.path.insert(0, "/home/aadivyar/csd-generation")
from synthesis.evaluate.benchmarks.sql_spider.dataset import load_spider

REPORT = "outputs/generated/spider7b_300x300_seed334_heldout_63pct_20260605/latest/results/success_report.json"
SPLIT = "environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json"
OUT_SPLIT = "environment/benchmark_splits/spider_oracle_struct_seed334.json"
OUT_REF = "/tmp/sc_oracle_reference.json"
N_WANT = 35

d = json.load(open(REPORT))
so = d["sample_outputs"]
split = json.load(open(SPLIT))
test_idx = split["test_indices"]
rows = load_spider(source="auto", indices=test_idx)
assert len(rows) == len(so) == len(test_idx) == 300

SQL_KW = set("""select from where group by order having limit as and or not in between like on
join inner left right outer full cross union intersect except distinct asc desc is null exists
all any case when then else end cast count sum avg min max abs round length lower upper substr
true false using natural""".split())

def schema_tokens(db_info):
    return set(re.findall(r"[a-z_][a-z0-9_]*", db_info.lower()))

def alias_names(sql):
    aliases = set()
    raw = re.findall(r"[a-z_][a-z0-9_]*", sql)
    for i, t in enumerate(raw):
        if t == "as" and i + 1 < len(raw):
            aliases.add(raw[i + 1])
    for m in re.finditer(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)\s+([a-z_][a-z0-9_]*)", sql):
        if m.group(2) not in SQL_KW:
            aliases.add(m.group(2))
    return aliases

def identifiers(sql):
    s = re.sub(r"'[^']*'|\"[^\"]*\"", " ", sql.lower())
    aliases = alias_names(s)
    return [t for t in re.findall(r"[a-z_][a-z0-9_]*", s)
            if t not in SQL_KW and t not in aliases and not re.fullmatch(r"t\d+", t)]

def in_schema(actual, db_info):
    schema = schema_tokens(db_info)
    return all(t in schema for t in identifiers(actual))

def top_level_select_list(sql):
    s = sql.strip(); low = s.lower(); si = low.find("select")
    if si < 0: return None
    depth = 0; i = si + 6; from_pos = -1
    while i < len(s):
        c = s[i]
        if c == "(": depth += 1
        elif c == ")": depth -= 1
        elif depth == 0 and low[i:i+4] == "from" and (i == 0 or not low[i-1].isalnum()):
            from_pos = i; break
        i += 1
    if from_pos < 0: return None
    inner = s[si+6:from_pos]
    parts, depth, cur = [], 0, ""
    for c in inner:
        if c == "(": depth += 1
        elif c == ")": depth -= 1
        if c == "," and depth == 0: parts.append(cur); cur = ""
        else: cur += c
    if cur.strip(): parts.append(cur)
    norm = []
    for p in parts:
        x = p.strip().lower()
        x = re.sub(r"\bas\s+[a-z_][a-z0-9_]*", "", x)
        x = re.sub(r"\b[a-z_][a-z0-9_]*\.", "", x)
        x = re.sub(r"\bdistinct\b", "", x)
        x = re.sub(r"\s+", "", x)
        if x: norm.append(x)
    return norm

select_same, select_diff = [], []
for pos, (r, e) in enumerate(zip(rows, so)):
    if e.get("is_correct") or not e.get("accuracy_applicable", True):
        continue
    actual = e.get("actual") or ""
    gold = e.get("expected") or ""
    if not in_schema(actual, r.get("db_info", "")):
        continue
    a = top_level_select_list(actual); g = top_level_select_list(gold)
    rec = {"pos": pos, "dev_index": test_idx[pos], "question": e.get("question", ""),
           "gold": gold, "argmax": actual}
    if a is not None and g is not None and a == g:
        select_same.append(rec)
    elif a is None or g is None or sorted(a or []) != sorted(g or []):
        select_diff.append(rec)

chosen = (select_same + select_diff)[:N_WANT]
chosen_dev = [c["dev_index"] for c in chosen]

probe = copy.deepcopy(split)
probe["test_indices"] = chosen_dev
probe["test_size"] = len(chosen_dev)
probe["test_preview"] = [c["question"][:60] for c in chosen]
json.dump(probe, open(OUT_SPLIT, "w"), indent=2)
json.dump({str(c["dev_index"]): c for c in chosen}, open(OUT_REF, "w"), indent=2)

print(f"select_same={len(select_same)} select_diff={len(select_diff)}")
print(f"chose {len(chosen)} structural-failure examples (dev indices): {chosen_dev}")
print(f"wrote split -> {OUT_SPLIT}")
print(f"wrote reference -> {OUT_REF}")
for c in chosen:
    print(f"  dev={c['dev_index']:>4}  Q: {c['question'][:55]}")
    print(f"        gold  : {c['gold'][:80]}")
    print(f"        argmax: {c['argmax'][:80]}")
