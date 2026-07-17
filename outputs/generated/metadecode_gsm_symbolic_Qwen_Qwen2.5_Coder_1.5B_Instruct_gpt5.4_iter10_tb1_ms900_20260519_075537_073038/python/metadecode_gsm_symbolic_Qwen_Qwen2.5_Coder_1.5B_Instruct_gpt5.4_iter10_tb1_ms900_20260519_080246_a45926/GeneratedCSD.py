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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. After the reasoning, give the final numeric answer inside << >> and keep only the final answer inside the delimiters.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openCount_: int
        d_2_openCount_ = 0
        out0_: int
        out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generatedPrefix, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        d_2_openCount_ = out0_
        d_3_forcedOpenDone_: bool
        d_3_forcedOpenDone_ = False
        if insideConstrained:
            d_3_forcedOpenDone_ = True
        elif (d_2_openCount_) > (0):
            d_3_forcedOpenDone_ = True
        d_4_initialChunkDone_: bool
        d_4_initialChunkDone_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_4_initialChunkDone_):
                            d_5_remaining0_: int
                            d_5_remaining0_ = (maxSteps) - (d_1_steps_)
                            d_6_chunkBudget_: int
                            if (d_5_remaining0_) > (24):
                                d_6_chunkBudget_ = 24
                            elif True:
                                d_6_chunkBudget_ = d_5_remaining0_
                            d_7_chunkedG_: _dafny.Seq
                            d_8_stoppedOpen_: bool
                            d_9_stoppedEos_: bool
                            d_10_stepsUsed_: int
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: bool
                            out4_: int
                            out1_, out2_, out3_, out4_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_7_chunkedG_ = out1_
                            d_8_stoppedOpen_ = out2_
                            d_9_stoppedEos_ = out3_
                            d_10_stepsUsed_ = out4_
                            generated = d_7_chunkedG_
                            d_1_steps_ = (d_1_steps_) + (d_10_stepsUsed_)
                            d_4_initialChunkDone_ = True
                            if d_9_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_8_stoppedOpen_:
                                d_11_enteredGenerated_: _dafny.Seq
                                d_12_enteredInside_: bool
                                d_13_enteredCurrent_: _dafny.Seq
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: _dafny.Seq
                                out5_, out6_, out7_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_11_enteredGenerated_ = out5_
                                d_12_enteredInside_ = out6_
                                d_13_enteredCurrent_ = out7_
                                generated = d_11_enteredGenerated_
                                insideConstrainedOut = d_12_enteredInside_
                                currentConstrainedOut = d_13_enteredCurrent_
                                d_3_forcedOpenDone_ = True
                        elif not(d_3_forcedOpenDone_):
                            d_14_openedGenerated_: _dafny.Seq
                            d_15_openedInside_: bool
                            d_16_openedCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_14_openedGenerated_ = out8_
                            d_15_openedInside_ = out9_
                            d_16_openedCurrent_ = out10_
                            generated = d_14_openedGenerated_
                            insideConstrainedOut = d_15_openedInside_
                            currentConstrainedOut = d_16_openedCurrent_
                            d_3_forcedOpenDone_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_17_remaining1_: int
                            d_17_remaining1_ = (maxSteps) - (d_1_steps_)
                            d_18_chunkBudget2_: int
                            if (d_17_remaining1_) > (8):
                                d_18_chunkBudget2_ = 8
                            elif True:
                                d_18_chunkBudget2_ = d_17_remaining1_
                            d_19_chunkedG2_: _dafny.Seq
                            d_20_stoppedOpen2_: bool
                            d_21_stoppedEos2_: bool
                            d_22_stepsUsed2_: int
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: bool
                            out14_: int
                            out11_, out12_, out13_, out14_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_18_chunkBudget2_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_19_chunkedG2_ = out11_
                            d_20_stoppedOpen2_ = out12_
                            d_21_stoppedEos2_ = out13_
                            d_22_stepsUsed2_ = out14_
                            generated = d_19_chunkedG2_
                            d_1_steps_ = (d_1_steps_) + (d_22_stepsUsed2_)
                            if d_21_stoppedEos2_:
                                raise _dafny.Break("0")
                            elif d_20_stoppedOpen2_:
                                d_23_enteredGenerated2_: _dafny.Seq
                                d_24_enteredInside2_: bool
                                d_25_enteredCurrent2_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_23_enteredGenerated2_ = out15_
                                d_24_enteredInside2_ = out16_
                                d_25_enteredCurrent2_ = out17_
                                generated = d_23_enteredGenerated2_
                                insideConstrainedOut = d_24_enteredInside2_
                                currentConstrainedOut = d_25_enteredCurrent2_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_26_closedGenerated_: _dafny.Seq
                        d_27_closedInside_: bool
                        d_28_closedCurrent_: _dafny.Seq
                        out18_: _dafny.Seq
                        out19_: bool
                        out20_: _dafny.Seq
                        out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_26_closedGenerated_ = out18_
                        d_27_closedInside_ = out19_
                        d_28_closedCurrent_ = out20_
                        generated = d_26_closedGenerated_
                        insideConstrainedOut = d_27_closedInside_
                        currentConstrainedOut = d_28_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_29_stablePrefix_: _dafny.Seq
                        d_29_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_30_constrainedPrompt_: _dafny.Seq
                        d_30_constrainedPrompt_ = (prompt) + (d_29_stablePrefix_)
                        d_31_validCount_: int
                        out21_: int
                        out21_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_31_validCount_ = out21_
                        d_32_next_: _dafny.Seq
                        d_32_next_ = eosToken
                        if (d_31_validCount_) <= (12):
                            out22_: _dafny.Seq
                            out22_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_30_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_32_next_ = out22_
                        elif True:
                            out23_: _dafny.Seq
                            out23_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_30_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                            d_32_next_ = out23_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_32_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_33_appendedGenerated_: _dafny.Seq
                            d_34_appendedInside_: bool
                            d_35_appendedCurrent_: _dafny.Seq
                            out24_: _dafny.Seq
                            out25_: bool
                            out26_: _dafny.Seq
                            out24_, out25_, out26_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_32_next_)
                            d_33_appendedGenerated_ = out24_
                            d_34_appendedInside_ = out25_
                            d_35_appendedCurrent_ = out26_
                            generated = d_33_appendedGenerated_
                            insideConstrainedOut = d_34_appendedInside_
                            currentConstrainedOut = d_35_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

