# start with gsm symbolic grammar from crane/dingo paper, test against gsm-symbolic, also folio
# https://github.com/apple/ml-gsm-symbolic
# write autosynthesized constrained decoder that can beat CRANE
# figure out which models are better, try QWEN-3 or Llama

# __main__.py
import module_ as module_
import _dafny as _dafny

from ConstrainedDecoding import default__

import os
import json

BENCH_PATH = "./datasets/ml-gsm-symbolic/generated_data/GSM_symbolic.jsonl"
OUT_DIR = "./outputs/ml-gsm-symbolic-outputs"
MAX_STEPS = 200

os.makedirs(OUT_DIR, exist_ok=True)

def load_gsm_list(path):
    """
    GSM_symbolic.jsonl contains ONE LINE which is a JSON list.
    We load that list and return it.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
        # If the file is a single JSON array, parse it directly.
        try:
            data = json.loads(raw)
            return data
        except json.JSONDecodeError:
            # fallback: treat as newline-delimited JSON objects
            data = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
            return data

def main():
    items = load_gsm_list(BENCH_PATH)
    results = []

    for idx, obj in enumerate(items):
        q = obj.get("question", "")
        prompt_text = f"""You are a math reasoning assistant. For the following word problem, produce step-by-step reasoning in normal text, and wrap each individual calculation or final expression in double angle brackets << >>.

Example format:
- Reasoning text here <<expression>> more reasoning <<expression>>... conclusion.

Problem:
{q}

Follow these rules:
1. Explain your reasoning in clear sentences.
2. Whenever you calculate a value or write a formula, wrap it in << >>.
3. Continue reasoning and interleaving calculations until the problem is fully solved.
4. Do not produce extra whitespace—use single spaces only.
5. Use the same << >> markers consistently for every expression.

Start your solution below:"""

        # call CRANE-like constrained decode
        solved = default__.ConstrainedDecode(prompt_text, MAX_STEPS)
        solved_expr = solved

        out_entry = {
            "id": obj.get("id"),
            "instance": obj.get("instance"),
            "question": q,
            "solution_expr": solved_expr
        }
        results.append(out_entry)

        print(out_entry)

    # final aggregate
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()