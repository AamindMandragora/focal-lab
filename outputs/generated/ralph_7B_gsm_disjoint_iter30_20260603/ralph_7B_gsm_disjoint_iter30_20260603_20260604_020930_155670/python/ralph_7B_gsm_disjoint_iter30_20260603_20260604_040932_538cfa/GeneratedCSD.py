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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve math word problems step by step. For each arithmetic step, write the expression inside << >> delimiters, for example <<3 + 4 = 7>>. Write the final answer as #### <<number>>. Every << must be immediately followed by a valid math expression and then >>. Do not leave << unclosed.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_constrainedTokenCount_: int
        d_2_constrainedTokenCount_ = 0
        d_3_maxConstrainedTokens_: int
        d_3_maxConstrainedTokens_ = 15
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_chunkBudget_: int
                        d_4_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_4_chunkBudget_) > (25):
                            d_4_chunkBudget_ = 25
                        if (d_4_chunkBudget_) == (0):
                            raise _dafny.Break("0")
                        d_5_chunkGenerated_: _dafny.Seq
                        d_6_stoppedOnOpenSpan_: bool
                        d_7_stoppedOnEos_: bool
                        d_8_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_5_chunkGenerated_ = out0_
                        d_6_stoppedOnOpenSpan_ = out1_
                        d_7_stoppedOnEos_ = out2_
                        d_8_stepsUsed_ = out3_
                        d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                        generated = d_5_chunkGenerated_
                        if d_7_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_6_stoppedOnOpenSpan_:
                            d_9_enteredGenerated_: _dafny.Seq
                            d_10_enteredInside_: bool
                            d_11_enteredCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_9_enteredGenerated_ = out4_
                            d_10_enteredInside_ = out5_
                            d_11_enteredCurrent_ = out6_
                            generated = d_9_enteredGenerated_
                            insideConstrainedOut = d_10_enteredInside_
                            currentConstrainedOut = d_11_enteredCurrent_
                            d_2_constrainedTokenCount_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        if ((d_1_steps_) + (1)) > (maxSteps):
                            raise _dafny.Break("0")
                        d_12_closedGenerated_: _dafny.Seq
                        d_13_closedInside_: bool
                        d_14_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_12_closedGenerated_ = out7_
                        d_13_closedInside_ = out8_
                        d_14_closedCurrent_ = out9_
                        generated = d_12_closedGenerated_
                        insideConstrainedOut = d_13_closedInside_
                        currentConstrainedOut = d_14_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_constrainedTokenCount_ = 0
                    elif (d_2_constrainedTokenCount_) >= (d_3_maxConstrainedTokens_):
                        d_15_rollbackSteps_: int
                        d_15_rollbackSteps_ = 0
                        d_16_maxRollback_: int
                        d_16_maxRollback_ = (d_3_maxConstrainedTokens_) + (2)
                        while ((d_15_rollbackSteps_) < (d_16_maxRollback_)) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                            d_17_rolledGenerated_: _dafny.Seq
                            d_18_rolledCurrent_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: _dafny.Seq
                            out10_, out11_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_17_rolledGenerated_ = out10_
                            d_18_rolledCurrent_ = out11_
                            generated = d_17_rolledGenerated_
                            currentConstrainedOut = d_18_rolledCurrent_
                            d_15_rollbackSteps_ = (d_15_rollbackSteps_) + (1)
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                            d_19_closedGenerated2_: _dafny.Seq
                            d_20_closedInside2_: bool
                            d_21_closedCurrent2_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_19_closedGenerated2_ = out12_
                            d_20_closedInside2_ = out13_
                            d_21_closedCurrent2_ = out14_
                            generated = d_19_closedGenerated2_
                            insideConstrainedOut = d_20_closedInside2_
                            currentConstrainedOut = d_21_closedCurrent2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_constrainedTokenCount_ = 0
                        elif True:
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                    elif True:
                        d_22_constrainedPrompt_: _dafny.Seq
                        d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_23_next_: _dafny.Seq
                        out15_: _dafny.Seq
                        out15_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_23_next_ = out15_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_23_next_) == (eosToken):
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                if ((d_1_steps_) + (1)) <= (maxSteps):
                                    d_24_closedGenerated3_: _dafny.Seq
                                    d_25_closedInside3_: bool
                                    d_26_closedCurrent3_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_24_closedGenerated3_ = out16_
                                    d_25_closedInside3_ = out17_
                                    d_26_closedCurrent3_ = out18_
                                    generated = d_24_closedGenerated3_
                                    insideConstrainedOut = d_25_closedInside3_
                                    currentConstrainedOut = d_26_closedCurrent3_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_2_constrainedTokenCount_ = 0
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                d_27_rbSteps_: int
                                d_27_rbSteps_ = 0
                                d_28_maxRb_: int
                                d_28_maxRb_ = (d_2_constrainedTokenCount_) + (2)
                                while ((d_27_rbSteps_) < (d_28_maxRb_)) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                                    d_29_rbGenerated_: _dafny.Seq
                                    d_30_rbCurrent_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out19_, out20_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_29_rbGenerated_ = out19_
                                    d_30_rbCurrent_ = out20_
                                    generated = d_29_rbGenerated_
                                    currentConstrainedOut = d_30_rbCurrent_
                                    d_27_rbSteps_ = (d_27_rbSteps_) + (1)
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                    d_31_closedGenerated4_: _dafny.Seq
                                    d_32_closedInside4_: bool
                                    d_33_closedCurrent4_: _dafny.Seq
                                    out21_: _dafny.Seq
                                    out22_: bool
                                    out23_: _dafny.Seq
                                    out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_31_closedGenerated4_ = out21_
                                    d_32_closedInside4_ = out22_
                                    d_33_closedCurrent4_ = out23_
                                    generated = d_31_closedGenerated4_
                                    insideConstrainedOut = d_32_closedInside4_
                                    currentConstrainedOut = d_33_closedCurrent4_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_2_constrainedTokenCount_ = 0
                                elif True:
                                    raise _dafny.Break("0")
                        elif True:
                            d_34_appendedGenerated_: _dafny.Seq
                            d_35_appendedInside_: bool
                            d_36_appendedCurrent_: _dafny.Seq
                            out24_: _dafny.Seq
                            out25_: bool
                            out26_: _dafny.Seq
                            out24_, out25_, out26_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                            d_34_appendedGenerated_ = out24_
                            d_35_appendedInside_ = out25_
                            d_36_appendedCurrent_ = out26_
                            generated = d_34_appendedGenerated_
                            insideConstrainedOut = d_35_appendedInside_
                            currentConstrainedOut = d_36_appendedCurrent_
                            d_2_constrainedTokenCount_ = (d_2_constrainedTokenCount_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

