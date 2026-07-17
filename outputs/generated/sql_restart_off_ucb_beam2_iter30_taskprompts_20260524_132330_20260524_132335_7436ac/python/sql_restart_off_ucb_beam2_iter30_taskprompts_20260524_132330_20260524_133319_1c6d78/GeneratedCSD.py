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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one line of the form: SQL: <<your single SQL query>>. The SQL query must appear inside the << and >> delimiters.")))
        if (not(insideConstrainedOut)) and ((cost) < (maxSteps)):
            d_1_chunkBudget_: int
            d_1_chunkBudget_ = (maxSteps) - (cost)
            if (d_1_chunkBudget_) > (16):
                d_1_chunkBudget_ = 16
            d_2_newGen_: _dafny.Seq
            d_3_stoppedOnOpen_: bool
            d_4_stoppedOnEos_: bool
            d_5_used_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_1_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_2_newGen_ = out0_
            d_3_stoppedOnOpen_ = out1_
            d_4_stoppedOnEos_ = out2_
            d_5_used_ = out3_
            generated = d_2_newGen_
            cost = (cost) + (d_5_used_)
            if d_3_stoppedOnOpen_:
                d_6_g2_: _dafny.Seq
                d_7_ic2_: bool
                d_8_cc2_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_6_g2_ = out4_
                d_7_ic2_ = out5_
                d_8_cc2_ = out6_
                generated = d_6_g2_
                insideConstrainedOut = d_7_ic2_
                currentConstrainedOut = d_8_cc2_
            elif (not(d_4_stoppedOnEos_)) and ((cost) < (maxSteps)):
                d_9_g3_: _dafny.Seq
                d_10_ic3_: bool
                d_11_cc3_: _dafny.Seq
                out7_: _dafny.Seq
                out8_: bool
                out9_: _dafny.Seq
                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_9_g3_ = out7_
                d_10_ic3_ = out8_
                d_11_cc3_ = out9_
                generated = d_9_g3_
                insideConstrainedOut = d_10_ic3_
                currentConstrainedOut = d_11_cc3_
                cost = (cost) + (1)
        with _dafny.label("0"):
            while (insideConstrainedOut) and ((cost) < (maxSteps)):
                with _dafny.c_label("0"):
                    d_12_isComplete_: bool
                    d_12_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                    if d_12_isComplete_:
                        raise _dafny.Break("0")
                    d_13_remaining_: int
                    d_13_remaining_ = (maxSteps) - (cost)
                    d_14_symBudget_: int
                    if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_13_remaining_)):
                        d_14_symBudget_ = d_13_remaining_
                    elif True:
                        d_14_symBudget_ = stepTokenBudget
                    if (d_14_symBudget_) == (0):
                        raise _dafny.Break("0")
                    d_15_stablePrefix_: _dafny.Seq
                    d_15_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                    d_16_constrainedPrompt_: _dafny.Seq
                    d_16_constrainedPrompt_ = (prompt) + (d_15_stablePrefix_)
                    d_17_newGen_: _dafny.Seq
                    d_18_newCur_: _dafny.Seq
                    d_19_hitEos_: bool
                    d_20_used_: int
                    out10_: _dafny.Seq
                    out11_: _dafny.Seq
                    out12_: bool
                    out13_: int
                    out10_, out11_, out12_, out13_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_16_constrainedPrompt_, generated, currentConstrainedOut, d_14_symBudget_, eosToken)
                    d_17_newGen_ = out10_
                    d_18_newCur_ = out11_
                    d_19_hitEos_ = out12_
                    d_20_used_ = out13_
                    generated = d_17_newGen_
                    currentConstrainedOut = d_18_newCur_
                    cost = (cost) + (d_20_used_)
                    if d_19_hitEos_:
                        raise _dafny.Break("0")
                    if (d_20_used_) == (0):
                        raise _dafny.Break("0")
                    pass
            pass
        if (insideConstrainedOut) and ((cost) < (maxSteps)):
            d_21_isComplete_: bool
            d_21_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if d_21_isComplete_:
                d_22_gC_: _dafny.Seq
                d_23_icC_: bool
                d_24_ccC_: _dafny.Seq
                out14_: _dafny.Seq
                out15_: bool
                out16_: _dafny.Seq
                out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_22_gC_ = out14_
                d_23_icC_ = out15_
                d_24_ccC_ = out16_
                generated = d_22_gC_
                insideConstrainedOut = d_23_icC_
                currentConstrainedOut = d_24_ccC_
                cost = (cost) + (1)
        return generated, insideConstrainedOut, currentConstrainedOut, cost

