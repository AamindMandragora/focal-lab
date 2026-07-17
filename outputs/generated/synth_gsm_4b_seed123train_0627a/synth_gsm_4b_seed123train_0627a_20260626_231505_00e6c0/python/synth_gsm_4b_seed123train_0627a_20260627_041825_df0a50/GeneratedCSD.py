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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "For intermediate steps and the final answer, write symbolic expressions inside << >>. Use only variable names (no {braces}), numbers, +, -, *, /, //, %, (, ). Example: <<n * price + 5>>. Always close with >>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_forcedOpen_: bool
        d_3_forcedOpen_ = False
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_remaining_: int
                        d_4_remaining_ = (maxSteps) - (d_2_steps_)
                        d_5_spanReserve_: int
                        d_5_spanReserve_ = 20
                        d_6_freeGenMax_: int
                        d_6_freeGenMax_ = 350
                        if ((d_4_remaining_) <= (d_5_spanReserve_)) or (d_3_forcedOpen_):
                            if (not(d_3_forcedOpen_)) and ((d_4_remaining_) >= (2)):
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: _dafny.Seq
                                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                generated = out0_
                                insideConstrainedOut = out1_
                                currentConstrainedOut = out2_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_3_forcedOpen_ = True
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_7_availForFree_: int
                            d_7_availForFree_ = (d_4_remaining_) - (d_5_spanReserve_)
                            d_8_chunkBudget_: int
                            if (d_7_availForFree_) > (d_6_freeGenMax_):
                                d_8_chunkBudget_ = d_6_freeGenMax_
                            elif True:
                                d_8_chunkBudget_ = d_7_availForFree_
                            d_9_chunkGenerated_: _dafny.Seq
                            d_10_stoppedOnOpenSpan_: bool
                            d_11_stoppedOnEos_: bool
                            d_12_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_9_chunkGenerated_ = out3_
                            d_10_stoppedOnOpenSpan_ = out4_
                            d_11_stoppedOnEos_ = out5_
                            d_12_stepsUsed_ = out6_
                            d_2_steps_ = (d_2_steps_) + (d_12_stepsUsed_)
                            generated = d_9_chunkGenerated_
                            if d_10_stoppedOnOpenSpan_:
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                generated = out7_
                                insideConstrainedOut = out8_
                                currentConstrainedOut = out9_
                            elif d_11_stoppedOnEos_:
                                d_13_rem2_: int
                                d_13_rem2_ = (maxSteps) - (d_2_steps_)
                                if (d_13_rem2_) >= (2):
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    generated = out10_
                                    insideConstrainedOut = out11_
                                    currentConstrainedOut = out12_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    d_3_forcedOpen_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                d_14_rem2_: int
                                d_14_rem2_ = (maxSteps) - (d_2_steps_)
                                if (d_14_rem2_) >= (2):
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    generated = out13_
                                    insideConstrainedOut = out14_
                                    currentConstrainedOut = out15_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    d_3_forcedOpen_ = True
                                elif True:
                                    raise _dafny.Break("0")
                    elif True:
                        d_15_remaining_: int
                        d_15_remaining_ = (maxSteps) - (d_2_steps_)
                        if (d_15_remaining_) == (0):
                            raise _dafny.Break("0")
                        d_16_spanCap_: int
                        d_16_spanCap_ = 80
                        d_17_closeBudget_: int
                        if (d_15_remaining_) > (d_16_spanCap_):
                            d_17_closeBudget_ = d_16_spanCap_
                        elif True:
                            d_17_closeBudget_ = d_15_remaining_
                        d_18_cg_: _dafny.Seq
                        d_19_ci_: bool
                        d_20_cc_: _dafny.Seq
                        out16_: _dafny.Seq
                        out17_: bool
                        out18_: _dafny.Seq
                        out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_closeBudget_)
                        d_18_cg_ = out16_
                        d_19_ci_ = out17_
                        d_20_cc_ = out18_
                        generated = d_18_cg_
                        insideConstrainedOut = d_19_ci_
                        currentConstrainedOut = d_20_cc_
                        d_2_steps_ = (d_2_steps_) + (d_17_closeBudget_)
                        d_3_forcedOpen_ = False
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

