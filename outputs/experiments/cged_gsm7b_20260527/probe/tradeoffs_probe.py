"""
Disposable probe: does the new trade-offs prompt section steer the author toward
gated/soft decoding when handed an accuracy-low / syntax-high strategy?

Inputs:
  - prev_strategy.txt: the control hard-masking GSM Dafny body (0.08 acc / 0.86 syntax)
  - task_description.txt: the GSM task description
Algorithm:
  1. Build the refinement prompt via build_evaluation_failure_prompt with the
     accuracy-low/syntax-high gap (acc 0.08 -> goal 0.25, syntax 0.86 -> goal 0.90),
     allowed_helpers=None so the full tool reference (incl. ConfidenceGatedStep) is shown.
  2. Call the configured large reasoning author once (bedrock / sonnet4.6).
  3. Save the returned Dafny body and report which decoding helpers it now uses.
Output:
  - probe_response.txt (full author output)
  - a printed verdict listing helper hits.
"""
import os
import sys

ROOT = "/home/aadivyar/csd-generation"
sys.path.insert(0, ROOT)

# Load the same credentials run_synthesis.py uses (bedrock bearer token, region, etc.)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, "synthesis", ".env"))

from synthesis.generate.prompts import build_evaluation_failure_prompt
from synthesis.generate.generator import StrategyGenerator

PROBE = os.path.join(ROOT, "outputs/experiments/cged_gsm7b_20260527/probe")

prev_strategy = open(os.path.join(PROBE, "prev_strategy.txt")).read()
task_description = open(os.path.join(PROBE, "task_description.txt")).read()

# Realistic feedback for the accuracy-low / syntax-high case: structure is valid
# almost everywhere, but the forced grammar-valid tokens push the model onto
# valid-but-wrong answers (the "mode_L" the prior rationale already named).
evaluation_feedback = (
    "Syntax is high (0.86) but accuracy is very low (0.08). The dominant failure is "
    "valid-but-wrong: outputs pass the grammar and contain a well-formed << >> answer "
    "span, yet the numeric answer is incorrect. The body hard-masks a grammar-valid token "
    "at every step inside the span (AppendConstrainedToken / ConstrainedStep), so when the "
    "model's preferred token is not grammar-valid it is forced onto a path it cannot recover "
    "from. The model's own reasoning is being overridden even where its top token would have "
    "been grammar-valid."
)

system_prompt, user_prompt = build_evaluation_failure_prompt(
    task_description=task_description,
    previous_strategy=prev_strategy,
    previous_accuracy=0.08,
    previous_syntax_rate=0.86,
    num_examples=50,
    goal_accuracy=0.25,
    goal_syntax_rate=0.90,
    evaluation_feedback=evaluation_feedback,
    allowed_helpers=None,
)

print("=== prompt sanity ===")
print("trade-offs section in user prompt:", "Trade-offs: matching constraint strength" in user_prompt)
print("ConfidenceGatedStep in user prompt:", "ConfidenceGatedStep" in user_prompt)
print("user prompt chars:", len(user_prompt))

gen = StrategyGenerator(
    backend="bedrock",
    model_name="us.anthropic.claude-sonnet-4-6",
    max_new_tokens=8000,
)
out = gen._generate_text(system_prompt, user_prompt)

with open(os.path.join(PROBE, "probe_response.txt"), "w") as f:
    f.write(out)

print("\n=== author response chars:", len(out), "===")
for h in ["ConfidenceGatedStep", "SoftConstrainedStep", "SafeSoftConstrainedStep",
          "AppendConstrainedToken", "ConstrainedStep", "SafeRepetitionPenaltyStep"]:
    print(f"  {h}: {out.count(h)}")
