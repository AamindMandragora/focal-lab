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
        (d_0_helpers_).AppendTaskGuidance(lm, ((((((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output format rules for expressions inside << >>: "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write bare Python expressions using ONLY the question's variable names with no curly braces. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Example: write 'm * x', NOT '{m} * {x}'; write 'n1 + n2', NOT '{n1} + {n2}'. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Allowed characters inside << >>: variable names (letters, digits, underscore), digits, and operators + - * / // % ( ). ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use // (floor division) when the answer must be a whole count of items, trips, people, or objects. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Wrap the final expression with int(...) when the answer is a whole-number quantity (e.g., int(n * frac_1 * frac_2)). ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use the exact variable spelling from the question (do not rename n1 to n_1 or vice versa). ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Do not use \\boxed, \\frac, $...$, or any LaTeX. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "End your response with: The final answer is <<EXPRESSION>>."))))
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

