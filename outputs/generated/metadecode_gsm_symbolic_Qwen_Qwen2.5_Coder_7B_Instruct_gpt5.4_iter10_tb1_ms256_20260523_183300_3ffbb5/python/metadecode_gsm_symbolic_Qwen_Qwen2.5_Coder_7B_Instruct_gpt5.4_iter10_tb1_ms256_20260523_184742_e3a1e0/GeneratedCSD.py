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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Write each arithmetic computation inside << >> delimiters.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_sawAnyOpen_: bool
        d_2_sawAnyOpen_ = insideConstrained
        d_3_forcedFirstOpen_: bool
        d_3_forcedFirstOpen_ = False
        d_4_preludeDone_: bool
        d_4_preludeDone_ = False
        d_5_narrowThreshold_: int
        d_5_narrowThreshold_ = 12
        if not(d_2_sawAnyOpen_):
            d_6_initialOpenCount_: int
            out0_: int
            out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generatedPrefix, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
            d_6_initialOpenCount_ = out0_
            if (d_6_initialOpenCount_) > (0):
                d_2_sawAnyOpen_ = True
                d_3_forcedFirstOpen_ = True
                d_4_preludeDone_ = True
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_4_preludeDone_):
                            d_7_remainingPrelude_: int
                            d_7_remainingPrelude_ = (maxSteps) - (d_1_steps_)
                            d_8_preludeBudget_: int
                            if (d_7_remainingPrelude_) > (8):
                                d_8_preludeBudget_ = 8
                            elif True:
                                d_8_preludeBudget_ = d_7_remainingPrelude_
                            d_9_chunkedG_: _dafny.Seq
                            d_10_stoppedOpen_: bool
                            d_11_stoppedEos_: bool
                            d_12_stepsUsed_: int
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: bool
                            out4_: int
                            out1_, out2_, out3_, out4_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_preludeBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_9_chunkedG_ = out1_
                            d_10_stoppedOpen_ = out2_
                            d_11_stoppedEos_ = out3_
                            d_12_stepsUsed_ = out4_
                            generated = d_9_chunkedG_
                            d_1_steps_ = (d_1_steps_) + (d_12_stepsUsed_)
                            d_4_preludeDone_ = True
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
                                d_2_sawAnyOpen_ = True
                        elif not(d_3_forcedFirstOpen_):
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
                            d_3_forcedFirstOpen_ = True
                            d_2_sawAnyOpen_ = True
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
                                    d_2_sawAnyOpen_ = True
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
                        d_28_validCount_: int
                        out18_: int
                        out18_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_28_validCount_ = out18_
                        d_29_nextIn_: _dafny.Seq
                        d_29_nextIn_ = eosToken
                        if (d_28_validCount_) <= (d_5_narrowThreshold_):
                            out19_: _dafny.Seq
                            out19_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_5_narrowThreshold_, eosToken)
                            d_29_nextIn_ = out19_
                        elif True:
                            d_30_nextSoft_: _dafny.Seq
                            d_31_usedFallback_: bool
                            out20_: _dafny.Seq
                            out21_: bool
                            out20_, out21_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_27_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                            d_30_nextSoft_ = out20_
                            d_31_usedFallback_ = out21_
                            d_29_nextIn_ = d_30_nextSoft_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_29_nextIn_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_32_appendedGenerated_: _dafny.Seq
                            d_33_appendedInside_: bool
                            d_34_appendedCurrent_: _dafny.Seq
                            out22_: _dafny.Seq
                            out23_: bool
                            out24_: _dafny.Seq
                            out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_29_nextIn_)
                            d_32_appendedGenerated_ = out22_
                            d_33_appendedInside_ = out23_
                            d_34_appendedCurrent_ = out24_
                            generated = d_32_appendedGenerated_
                            insideConstrainedOut = d_33_appendedInside_
                            currentConstrainedOut = d_34_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

