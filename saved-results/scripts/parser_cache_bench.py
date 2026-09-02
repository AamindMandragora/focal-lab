"""How much did dropping the prefix->text cache actually cost?"""
import os, sys, time
sys.path.insert(0, "/home/aadivyar/csd-generation")
os.chdir("/home/aadivyar/csd-generation")
grammar_text = open("synthesis/evaluate/grammars/smiles_chain_extenders.lark").read()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
tok.__class__ = type("CachedQwen2TokenizerFast", (type(tok),), {})
from synthesis.evaluate.benchmarks.common.parser_utils import create_lark_dafny_parser
sys.path.insert(0, "outputs/compiled_references/cars/ref_cars")
import _dafny, VerifiedDecoderAgent
from synthesis.evaluate.benchmarks.common.model_utils import _build_tokens_dafny
Parser = create_lark_dafny_parser(grammar_text, VerifiedDecoderAgent, _dafny,
    start="start", tokenizer=tok, accept_mask_backend="llguidance")
p = Parser(_build_tokens_dafny(_dafny, tok, list(range(len(tok)))))

def mk(n): return _dafny.SeqWithoutIsStrInference([_dafny.Seq("C") for _ in range(n)])

CALLS_PER_STEP = 5   # IsValidPrefix, IsCompletePrefix, ValidNextTokenCount, ValidNextToken, GroupHasValidMember

# what a cache hit would cost instead: one dict lookup
cache = {}
pre100 = mk(100)
cache[(id(pre100), len(pre100))] = (pre100, "x")
t0 = time.perf_counter()
for _ in range(200000):
    cache.get((id(pre100), len(pre100)))
hit_us = (time.perf_counter() - t0) / 200000 * 1e6

print(f"cache hit costs {hit_us:.3f} us\n")
print(f"{'answer len':>11}{'1 conversion (us)':>20}{'per step (ms)':>16}{'per example (s)':>18}{'50 examples (s)':>18}")
print("-" * 83)
for n in (50, 100, 200, 400, 800):
    pre = mk(n)
    reps = 2000
    t0 = time.perf_counter()
    for _ in range(reps):
        p._structured_text(pre)
    per_call = (time.perf_counter() - t0) / reps
    # generating an n-token answer walks lengths 1..n; cost is linear in length,
    # so the total is the per-call cost at n times n/2, times calls per step.
    per_example = per_call * (n / 2) * CALLS_PER_STEP
    print(f"{n:>11}{per_call*1e6:>20.1f}{per_call*CALLS_PER_STEP*1000:>16.3f}"
          f"{per_example:>18.3f}{per_example*50:>18.1f}")
print("-" * 83)
