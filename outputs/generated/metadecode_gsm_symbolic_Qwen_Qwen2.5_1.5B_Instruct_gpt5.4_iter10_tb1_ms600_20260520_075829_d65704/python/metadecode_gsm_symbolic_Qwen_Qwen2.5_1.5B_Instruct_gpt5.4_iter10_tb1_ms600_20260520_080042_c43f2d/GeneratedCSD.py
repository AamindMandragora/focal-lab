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
        if True:
            d_1_steps_: int
            d_1_steps_ = 0
            d_2_openedAtStart_: bool
            d_2_openedAtStart_ = False
            d_3_narrowThreshold_: int
            d_3_narrowThreshold_ = 12
            generated = generatedPrefix
            insideConstrainedOut = insideConstrained
            currentConstrainedOut = currentConstrained
            cost = 0
            if ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
                d_4_openedGenerated0_: _dafny.Seq
                d_5_openedInside0_: bool
                d_6_openedCurrent0_: _dafny.Seq
                out0_: _dafny.Seq
                out1_: bool
                out2_: _dafny.Seq
                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_4_openedGenerated0_ = out0_
                d_5_openedInside0_ = out1_
                d_6_openedCurrent0_ = out2_
                generated = d_4_openedGenerated0_
                insideConstrainedOut = d_5_openedInside0_
                currentConstrainedOut = d_6_openedCurrent0_
                d_2_openedAtStart_ = True
                d_1_steps_ = (d_1_steps_) + (1)
            with _dafny.label("0"):
                while (d_1_steps_) < (maxSteps):
                    with _dafny.c_label("0"):
                        if not(insideConstrainedOut):
                            d_7_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_7_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                                if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_8_closedGenerated_: _dafny.Seq
                                d_9_closedInside_: bool
                                d_10_closedCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_8_closedGenerated_ = out4_
                                d_9_closedInside_ = out5_
                                d_10_closedCurrent_ = out6_
                                generated = d_8_closedGenerated_
                                insideConstrainedOut = d_9_closedInside_
                                currentConstrainedOut = d_10_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_11_stablePrefix_: _dafny.Seq
                                d_11_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_12_constrainedPrompt_: _dafny.Seq
                                d_12_constrainedPrompt_ = (prompt) + (d_11_stablePrefix_)
                                d_13_validCount_: int
                                out7_: int
                                out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                d_13_validCount_ = out7_
                                if (d_13_validCount_) <= (d_3_narrowThreshold_):
                                    d_14_nextIn_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out8_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                                    d_14_nextIn_ = out8_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_14_nextIn_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_15_appendedGenerated_: _dafny.Seq
                                        d_16_appendedInside_: bool
                                        d_17_appendedCurrent_: _dafny.Seq
                                        out9_: _dafny.Seq
                                        out10_: bool
                                        out11_: _dafny.Seq
                                        out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_nextIn_)
                                        d_15_appendedGenerated_ = out9_
                                        d_16_appendedInside_ = out10_
                                        d_17_appendedCurrent_ = out11_
                                        generated = d_15_appendedGenerated_
                                        insideConstrainedOut = d_16_appendedInside_
                                        currentConstrainedOut = d_17_appendedCurrent_
                                elif True:
                                    d_18_remaining_: int
                                    d_18_remaining_ = (maxSteps) - (d_1_steps_)
                                    d_19_symbolBudget_: int
                                    if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_18_remaining_)):
                                        d_19_symbolBudget_ = d_18_remaining_
                                    elif True:
                                        d_19_symbolBudget_ = stepTokenBudget
                                    d_20_symbolGenerated_: _dafny.Seq
                                    d_21_symbolOut_: _dafny.Seq
                                    d_22_hitEos_: bool
                                    d_23_stepsUsed_: int
                                    out12_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: int
                                    out12_, out13_, out14_, out15_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_12_constrainedPrompt_, generated, currentConstrainedOut, d_19_symbolBudget_, eosToken)
                                    d_20_symbolGenerated_ = out12_
                                    d_21_symbolOut_ = out13_
                                    d_22_hitEos_ = out14_
                                    d_23_stepsUsed_ = out15_
                                    generated = d_20_symbolGenerated_
                                    insideConstrainedOut = True
                                    currentConstrainedOut = d_21_symbolOut_
                                    d_1_steps_ = (d_1_steps_) + (d_23_stepsUsed_)
                                    if d_22_hitEos_:
                                        raise _dafny.Break("0")
                        pass
                pass
            cost = d_1_steps_
            if ((((((maxSteps) > (0)) and ((cost) == (0))) and ((generated) == (generatedPrefix))) and ((insideConstrainedOut) == (insideConstrained))) and ((currentConstrainedOut) == (currentConstrained))) and (not(insideConstrainedOut)):
                d_24_openedGenerated1_: _dafny.Seq
                d_25_openedInside1_: bool
                d_26_openedCurrent1_: _dafny.Seq
                out16_: _dafny.Seq
                out17_: bool
                out18_: _dafny.Seq
                out16_, out17_, out18_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_24_openedGenerated1_ = out16_
                d_25_openedInside1_ = out17_
                d_26_openedCurrent1_ = out18_
                generated = d_24_openedGenerated1_
                insideConstrainedOut = d_25_openedInside1_
                currentConstrainedOut = d_26_openedCurrent1_
                cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

