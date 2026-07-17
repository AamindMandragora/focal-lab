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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve math word problems step by step. For each arithmetic step, write the expression inside << >> delimiters, for example <<3 + 4 = 7>>. Write the final answer as #### <<number>>. Every << must be immediately followed by a valid math expression and then >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_constrainedTokenCount_: int
        d_2_constrainedTokenCount_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_chunkBudget_: int
                        d_3_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_3_chunkBudget_) > (30):
                            d_3_chunkBudget_ = 30
                        if (d_3_chunkBudget_) == (0):
                            raise _dafny.Break("0")
                        d_4_chunkGenerated_: _dafny.Seq
                        d_5_stoppedOnOpenSpan_: bool
                        d_6_stoppedOnEos_: bool
                        d_7_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_4_chunkGenerated_ = out0_
                        d_5_stoppedOnOpenSpan_ = out1_
                        d_6_stoppedOnEos_ = out2_
                        d_7_stepsUsed_ = out3_
                        d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                        generated = d_4_chunkGenerated_
                        if d_6_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_5_stoppedOnOpenSpan_:
                            d_8_enteredGenerated_: _dafny.Seq
                            d_9_enteredInside_: bool
                            d_10_enteredCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_8_enteredGenerated_ = out4_
                            d_9_enteredInside_ = out5_
                            d_10_enteredCurrent_ = out6_
                            generated = d_8_enteredGenerated_
                            insideConstrainedOut = d_9_enteredInside_
                            currentConstrainedOut = d_10_enteredCurrent_
                            d_2_constrainedTokenCount_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        if ((d_1_steps_) + (1)) > (maxSteps):
                            raise _dafny.Break("0")
                        d_11_closedGenerated_: _dafny.Seq
                        d_12_closedInside_: bool
                        d_13_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_11_closedGenerated_ = out7_
                        d_12_closedInside_ = out8_
                        d_13_closedCurrent_ = out9_
                        generated = d_11_closedGenerated_
                        insideConstrainedOut = d_12_closedInside_
                        currentConstrainedOut = d_13_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_constrainedTokenCount_ = 0
                    elif (d_2_constrainedTokenCount_) >= (12):
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_14_rolledGenerated_: _dafny.Seq
                        d_15_rolledCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out10_, out11_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_14_rolledGenerated_ = out10_
                        d_15_rolledCurrent_ = out11_
                        generated = d_14_rolledGenerated_
                        currentConstrainedOut = d_15_rolledCurrent_
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                            d_16_closedGenerated2_: _dafny.Seq
                            d_17_closedInside2_: bool
                            d_18_closedCurrent2_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_16_closedGenerated2_ = out12_
                            d_17_closedInside2_ = out13_
                            d_18_closedCurrent2_ = out14_
                            generated = d_16_closedGenerated2_
                            insideConstrainedOut = d_17_closedInside2_
                            currentConstrainedOut = d_18_closedCurrent2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_constrainedTokenCount_ = 0
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        d_19_constrainedPrompt_: _dafny.Seq
                        d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_20_next_: _dafny.Seq
                        out15_: _dafny.Seq
                        out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_20_next_ = out15_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_20_next_) == (eosToken):
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                if ((d_1_steps_) + (1)) <= (maxSteps):
                                    d_21_closedGenerated3_: _dafny.Seq
                                    d_22_closedInside3_: bool
                                    d_23_closedCurrent3_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_21_closedGenerated3_ = out16_
                                    d_22_closedInside3_ = out17_
                                    d_23_closedCurrent3_ = out18_
                                    generated = d_21_closedGenerated3_
                                    insideConstrainedOut = d_22_closedInside3_
                                    currentConstrainedOut = d_23_closedCurrent3_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_2_constrainedTokenCount_ = 0
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                d_24_rbGenerated_: _dafny.Seq
                                d_25_rbCurrent_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: _dafny.Seq
                                out19_, out20_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_24_rbGenerated_ = out19_
                                d_25_rbCurrent_ = out20_
                                generated = d_24_rbGenerated_
                                currentConstrainedOut = d_25_rbCurrent_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                    d_26_closedGenerated4_: _dafny.Seq
                                    d_27_closedInside4_: bool
                                    d_28_closedCurrent4_: _dafny.Seq
                                    out21_: _dafny.Seq
                                    out22_: bool
                                    out23_: _dafny.Seq
                                    out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_26_closedGenerated4_ = out21_
                                    d_27_closedInside4_ = out22_
                                    d_28_closedCurrent4_ = out23_
                                    generated = d_26_closedGenerated4_
                                    insideConstrainedOut = d_27_closedInside4_
                                    currentConstrainedOut = d_28_closedCurrent4_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_2_constrainedTokenCount_ = 0
                                elif True:
                                    raise _dafny.Break("0")
                        elif True:
                            d_29_appendedGenerated_: _dafny.Seq
                            d_30_appendedInside_: bool
                            d_31_appendedCurrent_: _dafny.Seq
                            out24_: _dafny.Seq
                            out25_: bool
                            out26_: _dafny.Seq
                            out24_, out25_, out26_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                            d_29_appendedGenerated_ = out24_
                            d_30_appendedInside_ = out25_
                            d_31_appendedCurrent_ = out26_
                            generated = d_29_appendedGenerated_
                            insideConstrainedOut = d_30_appendedInside_
                            currentConstrainedOut = d_31_appendedCurrent_
                            d_2_constrainedTokenCount_ = (d_2_constrainedTokenCount_) + (1)
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                d_32_closedGenerated5_: _dafny.Seq
                                d_33_closedInside5_: bool
                                d_34_closedCurrent5_: _dafny.Seq
                                out27_: _dafny.Seq
                                out28_: bool
                                out29_: _dafny.Seq
                                out27_, out28_, out29_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_32_closedGenerated5_ = out27_
                                d_33_closedInside5_ = out28_
                                d_34_closedCurrent5_ = out29_
                                generated = d_32_closedGenerated5_
                                insideConstrainedOut = d_33_closedInside5_
                                currentConstrainedOut = d_34_closedCurrent5_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_constrainedTokenCount_ = 0
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

