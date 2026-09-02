"""CARS stop-rule probe.

focal's cars.dfy leaves the decode loop when
    parser.IsCompletePrefix(cur) && parser.ValidNextTokenCount(cur) == 0
Does that fire at the bare atom "C"? If yes, focal collapses every answer to
the shortest valid molecule -- the recorded 35/50 "<<C>>" failure.

Uses the REAL vocabulary, because the forbidden-token filter is sized to the
vocab and a short list makes the mask code throw (and a broad `except` then
returns an all-zero mask, which fakes exactly the answer we are looking for).

No model weights, no GPU. Grammar + tokenizer only.
"""
import os, sys, time
T0 = time.time()
def say(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)

sys.path.insert(0, "/home/aadivyar/csd-generation")
os.chdir("/home/aadivyar/csd-generation")

GRAMMAR = "synthesis/evaluate/grammars/smiles_chain_extenders.lark"
MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
grammar_text = open(GRAMMAR).read()

say("loading tokenizer (no model weights)")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)

# Real runs wrap the tokenizer in a caching subclass, and syncode keys its mask
# store directory on the class NAME. Adopt that name so we read the very store
# the real runs read, instead of triggering a 25-35 minute rebuild.
tok.__class__ = type("CachedQwen2TokenizerFast", (type(tok),), {})
say(f"tokenizer={type(tok).__name__} vocab_size={tok.vocab_size} len={len(tok)}")

from synthesis.evaluate.benchmarks.common.parser_utils import (
    _ensure_syncode_import_path, _get_parser_components, create_lark_dafny_parser,
)
_ensure_syncode_import_path()
import syncode.common as common
grammar, _bp, _lp, _clp = _get_parser_components(grammar_text, "start", complete_start=None)
dfa_path = (common.SYNCODE_CACHE + "mask_stores/" + type(tok).__name__ + "/"
            + f"grammar_mask_{grammar.hash()}_{tok.vocab_size}.pkl")
if not os.path.exists(dfa_path):
    say("dfa store MISSING -> 25-35 min to build. ABORTING, budget is 5 min.")
    sys.exit(3)
say(f"dfa store cached ({os.path.getsize(dfa_path)/1e6:.0f} MB)")

say("loading compiled Dafny reference")
sys.path.insert(0, "outputs/compiled_references/cars/ref_cars")
import _dafny
import VerifiedDecoderAgent

say("building parser (llguidance backend, exactly as SMILES does)")
Parser = create_lark_dafny_parser(
    grammar_text, VerifiedDecoderAgent, _dafny,
    start="start", tokenizer=tok, accept_mask_backend="llguidance",
)

from synthesis.evaluate.benchmarks.common.model_utils import _build_tokens_dafny
token_ids = list(range(len(tok)))
say(f"building real vocab token list, n={len(token_ids)}")
REAL_TOKENS = _build_tokens_dafny(_dafny, tok, token_ids)
p = Parser(REAL_TOKENS)
say("parser instance ready")

def mk(toks):
    return _dafny.SeqWithoutIsStrInference([_dafny.Seq(t) for t in toks])

CASES = [
    [],                                     # empty prefix (llguidance path)
    ["C"],                                  # the collapse answer
    ["C","C"],
    ["C","C","O"],
    ["O","C","C","O"],                      # a real diol chain extender
    ["C","1"],                              # incomplete ring
    ["C","1","C","C","C","C","C","1"],      # closed ring
    ["N"],
]

print("\n" + "="*82)
print(f"{'prefix':<26}{'IsComplete':<14}{'ValidNextCount':<18}{'focal break fires?'}")
print("="*82)
for toks in CASES:
    pre = mk(toks)
    txt = "".join(toks)
    try:
        comp = p.IsCompletePrefix(pre)
    except Exception as e:
        comp = f"ERR {type(e).__name__}"
    try:
        cnt = p.ValidNextTokenCount(pre)
    except Exception as e:
        cnt = f"ERR {type(e).__name__}"
    fires = (comp is True and cnt == 0)
    print(f"{txt!r:<26}{str(comp):<14}{str(cnt):<18}{'*** YES ***' if fires else 'no'}")
print("="*82)

# Is a zero count a real grammar verdict, or the broad `except` at
# parser_utils.py:413 swallowing a crash? Re-run the mask path with the catch
# removed so any real exception surfaces.
print("\nRaw mask path for 'C' with the broad except bypassed:")
try:
    r = p._inc_parser.get_acceptable_next_terminals("C")
    ms = p._dfa_mask_store
    say(f"  remainder={r.remainder!r}")
    if r.remainder is None:
        say("  remainder is None -> mask would be all-ones")
    else:
        st = ms._dfas.compute_dfa_states(r.remainder)
        m = ms._lookup_next_tokens(st, r)
        say(f"  dfa mask sum={int(m.sum())} len={m.numel()}")
        fb = p._forbidden_allow_mask
        say(f"  forbidden_allow_mask len={fb.numel()} sum={int(fb.sum())}")
        say(f"  after AND sum={int((m.cpu() & fb).sum()) if m.numel()==fb.numel() else 'SIZE MISMATCH'}")
except Exception as e:
    say(f"  RAISED {type(e).__name__}: {e}")
say("done")
