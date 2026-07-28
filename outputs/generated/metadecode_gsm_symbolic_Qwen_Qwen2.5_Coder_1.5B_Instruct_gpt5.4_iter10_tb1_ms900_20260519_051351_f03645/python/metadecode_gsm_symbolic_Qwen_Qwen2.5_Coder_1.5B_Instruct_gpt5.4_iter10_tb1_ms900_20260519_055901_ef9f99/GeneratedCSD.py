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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Put only the final numeric answer inside one final << >> span.")))
        if (maxSteps) == (0):
            cost = 0
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_answerSpanSeen_: bool
        d_2_answerSpanSeen_ = insideConstrained
        d_3_usedOutsideChunk_: bool
        d_3_usedOutsideChunk_ = False
        d_4_rollbackLimit_: int
        d_4_rollbackLimit_ = 24
        if not(insideConstrainedOut):
            d_5_openedGenerated0_: _dafny.Seq
            d_6_openedInside0_: bool
            d_7_openedCurrent0_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_5_openedGenerated0_ = out0_
            d_6_openedInside0_ = out1_
            d_7_openedCurrent0_ = out2_
            generated = d_5_openedGenerated0_
            insideConstrainedOut = d_6_openedInside0_
            currentConstrainedOut = d_7_openedCurrent0_
            d_2_answerSpanSeen_ = True
            d_1_steps_ = 1
        elif (parser).IsCompletePrefix(currentConstrainedOut):
            d_8_closedGenerated0_: _dafny.Seq
            d_9_closedInside0_: bool
            d_10_closedCurrent0_: _dafny.Seq
            out3_: _dafny.Seq
            out4_: bool
            out5_: _dafny.Seq
            out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
            d_8_closedGenerated0_ = out3_
            d_9_closedInside0_ = out4_
            d_10_closedCurrent0_ = out5_
            generated = d_8_closedGenerated0_
            insideConstrainedOut = d_9_closedInside0_
            currentConstrainedOut = d_10_closedCurrent0_
            d_1_steps_ = 1
        elif True:
            d_11_stablePrefix0_: _dafny.Seq
            d_11_stablePrefix0_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
            d_12_constrainedPrompt0_: _dafny.Seq
            d_12_constrainedPrompt0_ = (prompt) + (d_11_stablePrefix0_)
            d_13_next0_: _dafny.Seq
            out6_: _dafny.Seq
            out6_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_12_constrainedPrompt0_, currentConstrainedOut, eosToken)
            d_13_next0_ = out6_
            d_1_steps_ = 1
            if (d_13_next0_) != (eosToken):
                d_14_appendedGenerated0_: _dafny.Seq
                d_15_appendedInside0_: bool
                d_16_appendedCurrent0_: _dafny.Seq
                out7_: _dafny.Seq
                out8_: bool
                out9_: _dafny.Seq
                out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next0_)
                d_14_appendedGenerated0_ = out7_
                d_15_appendedInside0_ = out8_
                d_16_appendedCurrent0_ = out9_
                generated = d_14_appendedGenerated0_
                insideConstrainedOut = d_15_appendedInside0_
                currentConstrainedOut = d_16_appendedCurrent0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_3_usedOutsideChunk_)) and (not(d_2_answerSpanSeen_)):
                            d_17_remaining0_: int
                            d_17_remaining0_ = (maxSteps) - (d_1_steps_)
                            d_18_chunkBudget_: int
                            if (d_17_remaining0_) > (8):
                                d_18_chunkBudget_ = (d_17_remaining0_) - (8)
                            elif True:
                                d_18_chunkBudget_ = d_17_remaining0_
                            if (d_18_chunkBudget_) == (0):
                                if not(d_2_answerSpanSeen_):
                                    d_19_openedGenerated1_: _dafny.Seq
                                    d_20_openedInside1_: bool
                                    d_21_openedCurrent1_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_19_openedGenerated1_ = out10_
                                    d_20_openedInside1_ = out11_
                                    d_21_openedCurrent1_ = out12_
                                    generated = d_19_openedGenerated1_
                                    insideConstrainedOut = d_20_openedInside1_
                                    currentConstrainedOut = d_21_openedCurrent1_
                                    d_2_answerSpanSeen_ = True
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                d_22_chunkedG_: _dafny.Seq
                                d_23_stoppedOpen_: bool
                                d_24_stoppedEos_: bool
                                d_25_stepsUsed_: int
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: bool
                                out16_: int
                                out13_, out14_, out15_, out16_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_18_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                                d_22_chunkedG_ = out13_
                                d_23_stoppedOpen_ = out14_
                                d_24_stoppedEos_ = out15_
                                d_25_stepsUsed_ = out16_
                                generated = d_22_chunkedG_
                                d_1_steps_ = (d_1_steps_) + (d_25_stepsUsed_)
                                d_3_usedOutsideChunk_ = True
                                if d_24_stoppedEos_:
                                    if (not(d_2_answerSpanSeen_)) and ((d_1_steps_) < (maxSteps)):
                                        d_26_openedGenerated2_: _dafny.Seq
                                        d_27_openedInside2_: bool
                                        d_28_openedCurrent2_: _dafny.Seq
                                        out17_: _dafny.Seq
                                        out18_: bool
                                        out19_: _dafny.Seq
                                        out17_, out18_, out19_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                        d_26_openedGenerated2_ = out17_
                                        d_27_openedInside2_ = out18_
                                        d_28_openedCurrent2_ = out19_
                                        generated = d_26_openedGenerated2_
                                        insideConstrainedOut = d_27_openedInside2_
                                        currentConstrainedOut = d_28_openedCurrent2_
                                        d_2_answerSpanSeen_ = True
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        raise _dafny.Break("0")
                                elif d_23_stoppedOpen_:
                                    d_29_enteredGenerated_: _dafny.Seq
                                    d_30_enteredInside_: bool
                                    d_31_enteredCurrent_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out21_: bool
                                    out22_: _dafny.Seq
                                    out20_, out21_, out22_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_29_enteredGenerated_ = out20_
                                    d_30_enteredInside_ = out21_
                                    d_31_enteredCurrent_ = out22_
                                    generated = d_29_enteredGenerated_
                                    insideConstrainedOut = d_30_enteredInside_
                                    currentConstrainedOut = d_31_enteredCurrent_
                                    d_2_answerSpanSeen_ = True
                                elif (not(d_2_answerSpanSeen_)) and ((d_1_steps_) < (maxSteps)):
                                    d_32_openedGenerated3_: _dafny.Seq
                                    d_33_openedInside3_: bool
                                    d_34_openedCurrent3_: _dafny.Seq
                                    out23_: _dafny.Seq
                                    out24_: bool
                                    out25_: _dafny.Seq
                                    out23_, out24_, out25_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_32_openedGenerated3_ = out23_
                                    d_33_openedInside3_ = out24_
                                    d_34_openedCurrent3_ = out25_
                                    generated = d_32_openedGenerated3_
                                    insideConstrainedOut = d_33_openedInside3_
                                    currentConstrainedOut = d_34_openedCurrent3_
                                    d_2_answerSpanSeen_ = True
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                        elif not(d_2_answerSpanSeen_):
                            d_35_openedGenerated4_: _dafny.Seq
                            d_36_openedInside4_: bool
                            d_37_openedCurrent4_: _dafny.Seq
                            out26_: _dafny.Seq
                            out27_: bool
                            out28_: _dafny.Seq
                            out26_, out27_, out28_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_35_openedGenerated4_ = out26_
                            d_36_openedInside4_ = out27_
                            d_37_openedCurrent4_ = out28_
                            generated = d_35_openedGenerated4_
                            insideConstrainedOut = d_36_openedInside4_
                            currentConstrainedOut = d_37_openedCurrent4_
                            d_2_answerSpanSeen_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_38_closedGenerated_: _dafny.Seq
                        d_39_closedInside_: bool
                        d_40_closedCurrent_: _dafny.Seq
                        out29_: _dafny.Seq
                        out30_: bool
                        out31_: _dafny.Seq
                        out29_, out30_, out31_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_38_closedGenerated_ = out29_
                        d_39_closedInside_ = out30_
                        d_40_closedCurrent_ = out31_
                        generated = d_38_closedGenerated_
                        insideConstrainedOut = d_39_closedInside_
                        currentConstrainedOut = d_40_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif (len(currentConstrainedOut)) >= (d_4_rollbackLimit_):
                        d_41_rolledGenerated_: _dafny.Seq
                        d_42_rolledCurrent_: _dafny.Seq
                        out32_: _dafny.Seq
                        out33_: _dafny.Seq
                        out32_, out33_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_41_rolledGenerated_ = out32_
                        d_42_rolledCurrent_ = out33_
                        generated = d_41_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_42_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_43_stablePrefix_: _dafny.Seq
                        d_43_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_44_constrainedPrompt_: _dafny.Seq
                        d_44_constrainedPrompt_ = (prompt) + (d_43_stablePrefix_)
                        d_45_next_: _dafny.Seq
                        out34_: _dafny.Seq
                        out34_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_44_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_45_next_ = out34_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_45_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_46_appendedGenerated_: _dafny.Seq
                            d_47_appendedInside_: bool
                            d_48_appendedCurrent_: _dafny.Seq
                            out35_: _dafny.Seq
                            out36_: bool
                            out37_: _dafny.Seq
                            out35_, out36_, out37_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_45_next_)
                            d_46_appendedGenerated_ = out35_
                            d_47_appendedInside_ = out36_
                            d_48_appendedCurrent_ = out37_
                            generated = d_46_appendedGenerated_
                            insideConstrainedOut = d_47_appendedInside_
                            currentConstrainedOut = d_48_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

