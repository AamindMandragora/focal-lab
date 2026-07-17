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
        (d_0_helpers_).AppendTaskGuidance(lm, (((((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Format rules for your answer: "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(1) Wrap every intermediate arithmetic expression and the final answer in << ... >> delimiters. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(2) Never use LaTeX such as \\boxed{}, \\frac{}, \\(, \\[, or $...$. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(3) Inside << ... >> use only basic arithmetic: + - * / // ( ) and the given variable names. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Do NOT use min, max, abs, round, if/else, conditionals, words, or quotes inside << >>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(4) Always simplify the algebra before writing the final answer; combine like terms. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(5) End your response with a single line of the form: The final answer is <<EXPRESSION>>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use the exact variable names from the question (e.g. n, m, x, n_1) without braces."))))
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

