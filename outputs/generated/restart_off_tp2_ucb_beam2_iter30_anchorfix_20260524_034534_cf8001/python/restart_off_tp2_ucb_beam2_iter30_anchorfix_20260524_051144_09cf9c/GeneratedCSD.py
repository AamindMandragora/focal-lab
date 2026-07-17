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
        if (maxSteps) == (0):
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You are solving the SPECIFIC symbolic math word problem in the user message above. The problem uses variables in curly braces like {n}, {x}, {name}, {unit}, {sides}, {target}. Inside any math expression use the BARE variable name without braces (write n, not {n}).\n\nCRITICAL ANTI-HALLUCINATION: Some canonical GSM problems are heavily memorized. You must NOT default to them. Do NOT write any of the following unless they literally appear in THIS user problem: 'Initially, there are {t} trees', 'there are {c} cars in the parking lot', '{nc} more cars arrive', 'workers will plant trees', 'tf - t', 'c + nc'. If you find yourself writing those phrases, STOP — you are solving the wrong problem.\n\nProcedure: (A) First, in one short sentence, list ONLY the variable names that literally appear inside curly braces in the user's question. (B) Then briefly explain the calculation using only those variables. (C) End with EXACTLY ONE << final_formula >> on the last line and stop. Do NOT emit any intermediate << >> spans; the final formula is the only << >>.\n\nFormula rules: (1) Use ONLY variable names from step (A); never invent new ones. (2) For whole-count answers, use // (Python integer division), not /. (3) For percentage answers, multiply by 100 and wrap with int(...). (4) Unit conversions: 1 foot=12 inches, 1 hour=60 minutes, 1 year=12 months, 1 day=24 hours, 1 pound=16 ounces. (5) Do NOT add ceiling-division glue like '+ int(x % y > 0)'; write the direct formula the question asks for. (6) Mirror the problem's stated order of operations; do not algebraically rewrite. If it says 'rate t per chunk d, total y', write y//d*t (chunks first, then rate), not y*t/d. (7) When a fractional/decimal variable multiplies an integer to produce a whole-count subterm, wrap ONLY that specific product in int(...) and keep other terms bare: write n - n1*w1 - int(n3*w3), not int(n - n1*w1 - n3*w3). (8) Re-read the final question sentence to identify which named quantity is the answer.\n\nNow solve THIS problem (not a memorized one):")))
        while (cost) < (maxSteps):
            d_1_remaining_: int
            d_1_remaining_ = (maxSteps) - (cost)
            d_2_newGenerated_: _dafny.Seq
            d_3_stoppedOnOpenSpan_: bool
            d_4_stoppedOnEos_: bool
            d_5_stepsUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_1_remaining_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_2_newGenerated_ = out0_
            d_3_stoppedOnOpenSpan_ = out1_
            d_4_stoppedOnEos_ = out2_
            d_5_stepsUsed_ = out3_
            generated = d_2_newGenerated_
            cost = (cost) + (d_5_stepsUsed_)
            if d_4_stoppedOnEos_:
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            if (d_5_stepsUsed_) == (0):
                return generated, insideConstrainedOut, currentConstrainedOut, cost
        return generated, insideConstrainedOut, currentConstrainedOut, cost

