import sys
from typing import Callable, Any, TypeVar, NamedTuple
from math import floor
from itertools import count

import module_ as module_
import _dafny as _dafny
import System_ as System_
import VerifiedDecoderAgent as VerifiedDecoderAgent

# Module: GeneratedCSD

class default__:
    def  __init__(self):
        pass

    @staticmethod
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
        generated: _dafny.Seq = _dafny.Seq({})
        insideConstrainedOut: bool = False
        currentConstrainedOut: _dafny.Seq = _dafny.Seq({})
        cost: int = int(0)
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__()
        d_0_helpers_ = nw0_
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        (d_0_helpers_).AppendTaskGuidance(lm, (((((((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output format: The question uses template placeholders like {n}, {m}, {frac_1}. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "In YOUR answer, write those as bare identifiers (n, m, frac_1) — never include the characters { or } anywhere in your output. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Never use LaTeX: no \\boxed, no \\frac, no \\[, no $...$, no \\text. Plain text only. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Format example: 'Let's think step by step. <reasoning using bare variable names>. The total is <<n + m>>. The final answer is <<n + m>>.' ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Math rules: ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use the SAME variable spelling as the question (do not rewrite n1 as n_1 or vice versa). ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use // (Python floor division) whenever the answer must be a whole count of items, trips, people, or objects; reserve / only when a fractional value is intended. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Do not algebraically simplify or combine terms across steps — leave each step's expression in the same form as the corresponding calculation in the problem, and write the final answer as a direct expression of those original quantities. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write every computed expression inside << ... >> using only + - * / // ( ) and the question's variables. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "End with: The final answer is <<EXPRESSION>>."))))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_stopped_: bool
        d_2_stopped_ = False
        while ((d_1_steps_) < (maxSteps)) and (not(d_2_stopped_)):
            d_3_tok_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_3_tok_ = out0_
            d_1_steps_ = (d_1_steps_) + (1)
            cost = (cost) + (1)
            if (d_3_tok_) == (eosToken):
                d_2_stopped_ = True
            elif True:
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_tok_]))
        return generated, insideConstrainedOut, currentConstrainedOut, cost

