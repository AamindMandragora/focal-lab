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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one SQL query in the required constrained span and avoid extra explanation.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openedByStrategy_: bool
        d_2_openedByStrategy_ = False
        d_3_usedPrelude_: bool
        d_3_usedPrelude_ = False
        d_4_narrowThreshold_: int
        d_4_narrowThreshold_ = 12
        d_5_preludeBudget_: int
        d_5_preludeBudget_ = 4
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_openCount_: int
                        out0_: int
                        out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                        d_6_openCount_ = out0_
                        if ((not(d_3_usedPrelude_)) and (not(d_2_openedByStrategy_))) and ((d_6_openCount_) == (0)):
                            d_7_remainingPrelude_: int
                            d_7_remainingPrelude_ = (maxSteps) - (d_1_steps_)
                            d_8_chunkBudget_: int
                            if (d_5_preludeBudget_) > (d_7_remainingPrelude_):
                                d_8_chunkBudget_ = d_7_remainingPrelude_
                            elif True:
                                d_8_chunkBudget_ = d_5_preludeBudget_
                            d_9_chunkedGenerated_: _dafny.Seq
                            d_10_stoppedOpen_: bool
                            d_11_stoppedEos_: bool
                            d_12_stepsUsed_: int
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: bool
                            out4_: int
                            out1_, out2_, out3_, out4_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_9_chunkedGenerated_ = out1_
                            d_10_stoppedOpen_ = out2_
                            d_11_stoppedEos_ = out3_
                            d_12_stepsUsed_ = out4_
                            generated = d_9_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_12_stepsUsed_)
                            d_3_usedPrelude_ = True
                            if d_11_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_10_stoppedOpen_:
                                d_13_enteredGenerated_: _dafny.Seq
                                d_14_enteredInside_: bool
                                d_15_enteredCurrent_: _dafny.Seq
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: _dafny.Seq
                                out5_, out6_, out7_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_13_enteredGenerated_ = out5_
                                d_14_enteredInside_ = out6_
                                d_15_enteredCurrent_ = out7_
                                generated = d_13_enteredGenerated_
                                insideConstrainedOut = d_14_enteredInside_
                                currentConstrainedOut = d_15_enteredCurrent_
                        elif (not(d_2_openedByStrategy_)) and ((d_6_openCount_) == (0)):
                            d_16_openedGenerated_: _dafny.Seq
                            d_17_openedInside_: bool
                            d_18_openedCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_16_openedGenerated_ = out8_
                            d_17_openedInside_ = out9_
                            d_18_openedCurrent_ = out10_
                            generated = d_16_openedGenerated_
                            insideConstrainedOut = d_17_openedInside_
                            currentConstrainedOut = d_18_openedCurrent_
                            d_2_openedByStrategy_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_19_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_19_next_ = out11_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_19_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_19_next_]))
                                if (d_19_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_20_enteredGenerated2_: _dafny.Seq
                                    d_21_enteredInside2_: bool
                                    d_22_enteredCurrent2_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_20_enteredGenerated2_ = out12_
                                    d_21_enteredInside2_ = out13_
                                    d_22_enteredCurrent2_ = out14_
                                    generated = d_20_enteredGenerated2_
                                    insideConstrainedOut = d_21_enteredInside2_
                                    currentConstrainedOut = d_22_enteredCurrent2_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_23_closedGenerated_: _dafny.Seq
                        d_24_closedInside_: bool
                        d_25_closedCurrent_: _dafny.Seq
                        out15_: _dafny.Seq
                        out16_: bool
                        out17_: _dafny.Seq
                        out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_23_closedGenerated_ = out15_
                        d_24_closedInside_ = out16_
                        d_25_closedCurrent_ = out17_
                        generated = d_23_closedGenerated_
                        insideConstrainedOut = d_24_closedInside_
                        currentConstrainedOut = d_25_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_26_stablePrefix_: _dafny.Seq
                        d_26_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_27_constrainedPrompt_: _dafny.Seq
                        d_27_constrainedPrompt_ = (prompt) + (d_26_stablePrefix_)
                        d_28_next_: _dafny.Seq
                        out18_: _dafny.Seq
                        out18_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_4_narrowThreshold_, eosToken)
                        d_28_next_ = out18_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_28_next_) == (eosToken):
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_29_closedGenerated2_: _dafny.Seq
                                d_30_closedInside2_: bool
                                d_31_closedCurrent2_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_29_closedGenerated2_ = out19_
                                d_30_closedInside2_ = out20_
                                d_31_closedCurrent2_ = out21_
                                generated = d_29_closedGenerated2_
                                insideConstrainedOut = d_30_closedInside2_
                                currentConstrainedOut = d_31_closedCurrent2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_32_rolledGenerated_: _dafny.Seq
                                d_33_rolledCurrent_: _dafny.Seq
                                out22_: _dafny.Seq
                                out23_: _dafny.Seq
                                out22_, out23_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_32_rolledGenerated_ = out22_
                                d_33_rolledCurrent_ = out23_
                                generated = d_32_rolledGenerated_
                                insideConstrainedOut = True
                                currentConstrainedOut = d_33_rolledCurrent_
                                if (len(d_33_rolledCurrent_)) == (len(currentConstrainedOut)):
                                    raise _dafny.Break("0")
                        elif True:
                            d_34_appendedGenerated_: _dafny.Seq
                            d_35_appendedInside_: bool
                            d_36_appendedCurrent_: _dafny.Seq
                            out24_: _dafny.Seq
                            out25_: bool
                            out26_: _dafny.Seq
                            out24_, out25_, out26_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_28_next_)
                            d_34_appendedGenerated_ = out24_
                            d_35_appendedInside_ = out25_
                            d_36_appendedCurrent_ = out26_
                            generated = d_34_appendedGenerated_
                            insideConstrainedOut = d_35_appendedInside_
                            currentConstrainedOut = d_36_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

