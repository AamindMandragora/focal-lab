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
        out0_: _dafny.Seq
        out1_: bool
        out2_: _dafny.Seq
        out3_: int
        out0_, out1_, out2_, out3_ = default__.AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken)
        generated = out0_
        insideConstrainedOut = out1_
        currentConstrainedOut = out2_
        cost = out3_
        if ((maxSteps) > (0)) and ((cost) <= (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

    @staticmethod
    def AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        if (maxSteps) == (0):
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly: SQL: <<SELECT ...>>. Write only valid SQL inside the angle brackets. Use exact table and column names from the schema. No explanation."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_prefixBudget_: int
        d_2_prefixBudget_ = 8
        if (d_2_prefixBudget_) > (maxSteps):
            d_2_prefixBudget_ = maxSteps
        d_3_boostAmount_: _dafny.BigRational
        d_3_boostAmount_ = _dafny.BigRational('4e0')
        d_4_narrowThreshold_: int
        d_4_narrowThreshold_ = 16
        d_5_phase1Budget_: int
        d_5_phase1Budget_ = _dafny.euclidian_division(maxSteps, 2)
        if (d_5_phase1Budget_) < (d_2_prefixBudget_):
            d_5_phase1Budget_ = d_2_prefixBudget_
        if (d_5_phase1Budget_) > (maxSteps):
            d_5_phase1Budget_ = maxSteps
        d_6_gOut_: _dafny.Seq
        d_7_iOut_: bool
        d_8_cOut_: _dafny.Seq
        out0_: _dafny.Seq
        out1_: bool
        out2_: _dafny.Seq
        out0_, out1_, out2_ = (d_0_helpers_).GenerateWithPrefixAndManagedSpan(lm, parser, prompt, generated, insideConstrainedOut, currentConstrainedOut, d_5_phase1Budget_, d_2_prefixBudget_, validTokenGroups, d_3_boostAmount_, d_4_narrowThreshold_, eosToken)
        d_6_gOut_ = out0_
        d_7_iOut_ = out1_
        d_8_cOut_ = out2_
        generated = d_6_gOut_
        insideConstrainedOut = d_7_iOut_
        currentConstrainedOut = d_8_cOut_
        if ((insideConstrainedOut) and ((d_5_phase1Budget_) < (maxSteps))) and ((len(currentConstrainedOut)) <= (len(generated))):
            d_9_remainingBudget_: int
            d_9_remainingBudget_ = (maxSteps) - (d_5_phase1Budget_)
            d_10_cg_: _dafny.Seq
            d_11_ci_: bool
            d_12_cc_: _dafny.Seq
            out3_: _dafny.Seq
            out4_: bool
            out5_: _dafny.Seq
            out3_, out4_, out5_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_9_remainingBudget_)
            d_10_cg_ = out3_
            d_11_ci_ = out4_
            d_12_cc_ = out5_
            generated = d_10_cg_
            insideConstrainedOut = d_11_ci_
            currentConstrainedOut = d_12_cc_
        cost = maxSteps
        return generated, insideConstrainedOut, currentConstrainedOut, cost

