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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step using the variable names from the problem (like n1, n2, k, x, t, etc.). Write your reasoning in plain text. At the very end of your solution, write the complete final answer expression inside << >> delimiters. Use only ONE pair of << >> for the final answer. The expression inside << >> must be complete and use the symbolic variable names from the problem. Example: <<count * (n1 + n2 + n3)>>. Do not truncate the expression.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_maxConstrainedLen_: int
        d_2_maxConstrainedLen_ = 60
        d_3_hasOpenedSpan_: bool
        d_3_hasOpenedSpan_ = insideConstrained
        d_4_rollbackCount_: int
        d_4_rollbackCount_ = 0
        d_5_maxRollbacks_: int
        d_5_maxRollbacks_ = 3
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_3_hasOpenedSpan_)) and ((d_1_steps_) >= ((maxSteps) - (5))):
                            d_6_openedGenerated_: _dafny.Seq
                            d_7_openedInside_: bool
                            d_8_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_openedGenerated_ = out0_
                            d_7_openedInside_ = out1_
                            d_8_openedCurrent_ = out2_
                            generated = d_6_openedGenerated_
                            insideConstrainedOut = d_7_openedInside_
                            currentConstrainedOut = d_8_openedCurrent_
                            d_3_hasOpenedSpan_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_chunkBudget_: int
                            d_9_chunkBudget_ = (maxSteps) - (d_1_steps_)
                            d_10_chunkedG_: _dafny.Seq
                            d_11_stoppedOpen_: bool
                            d_12_stoppedEos_: bool
                            d_13_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_9_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_10_chunkedG_ = out3_
                            d_11_stoppedOpen_ = out4_
                            d_12_stoppedEos_ = out5_
                            d_13_stepsUsed_ = out6_
                            generated = d_10_chunkedG_
                            d_1_steps_ = (d_1_steps_) + (d_13_stepsUsed_)
                            if d_11_stoppedOpen_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_3_hasOpenedSpan_ = True
                                d_4_rollbackCount_ = 0
                            elif d_12_stoppedEos_:
                                if (not(d_3_hasOpenedSpan_)) and ((d_1_steps_) < (maxSteps)):
                                    d_14_openedGenerated_: _dafny.Seq
                                    d_15_openedInside_: bool
                                    d_16_openedCurrent_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_14_openedGenerated_ = out7_
                                    d_15_openedInside_ = out8_
                                    d_16_openedCurrent_ = out9_
                                    generated = d_14_openedGenerated_
                                    insideConstrainedOut = d_15_openedInside_
                                    currentConstrainedOut = d_16_openedCurrent_
                                    d_3_hasOpenedSpan_ = True
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_17_closedGenerated_: _dafny.Seq
                        d_18_closedInside_: bool
                        d_19_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_17_closedGenerated_ = out10_
                        d_18_closedInside_ = out11_
                        d_19_closedCurrent_ = out12_
                        generated = d_17_closedGenerated_
                        insideConstrainedOut = d_18_closedInside_
                        currentConstrainedOut = d_19_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_4_rollbackCount_ = 0
                    elif ((len(currentConstrainedOut)) >= (d_2_maxConstrainedLen_)) or ((d_4_rollbackCount_) >= (d_5_maxRollbacks_)):
                        insideConstrainedOut = False
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_20_constrainedPrompt_: _dafny.Seq
                        d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_21_next_: _dafny.Seq
                        out13_: _dafny.Seq
                        out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_21_next_ = out13_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_21_next_) == (eosToken):
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                pass
                            elif True:
                                d_22_rolledGenerated_: _dafny.Seq
                                d_23_rolledCurrent_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: _dafny.Seq
                                out14_, out15_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_22_rolledGenerated_ = out14_
                                d_23_rolledCurrent_ = out15_
                                generated = d_22_rolledGenerated_
                                currentConstrainedOut = d_23_rolledCurrent_
                                d_4_rollbackCount_ = (d_4_rollbackCount_) + (1)
                                if ((parser).IsCompletePrefix(d_23_rolledCurrent_)) and ((d_1_steps_) < (maxSteps)):
                                    d_24_closedGenerated_: _dafny.Seq
                                    d_25_closedInside_: bool
                                    d_26_closedCurrent_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, d_23_rolledCurrent_)
                                    d_24_closedGenerated_ = out16_
                                    d_25_closedInside_ = out17_
                                    d_26_closedCurrent_ = out18_
                                    generated = d_24_closedGenerated_
                                    insideConstrainedOut = d_25_closedInside_
                                    currentConstrainedOut = d_26_closedCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                        elif True:
                            d_27_appendedGenerated_: _dafny.Seq
                            d_28_appendedInside_: bool
                            d_29_appendedCurrent_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: _dafny.Seq
                            out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                            d_27_appendedGenerated_ = out19_
                            d_28_appendedInside_ = out20_
                            d_29_appendedCurrent_ = out21_
                            generated = d_27_appendedGenerated_
                            insideConstrainedOut = d_28_appendedInside_
                            currentConstrainedOut = d_29_appendedCurrent_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_30_closedGenerated_: _dafny.Seq
                d_31_closedInside_: bool
                d_32_closedCurrent_: _dafny.Seq
                out22_: _dafny.Seq
                out23_: bool
                out24_: _dafny.Seq
                out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_30_closedGenerated_ = out22_
                d_31_closedInside_ = out23_
                d_32_closedCurrent_ = out24_
                generated = d_30_closedGenerated_
                insideConstrainedOut = d_31_closedInside_
                currentConstrainedOut = d_32_closedCurrent_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_33_rolledGenerated_: _dafny.Seq
                d_34_rolledCurrent_: _dafny.Seq
                out25_: _dafny.Seq
                out26_: _dafny.Seq
                out25_, out26_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_33_rolledGenerated_ = out25_
                d_34_rolledCurrent_ = out26_
                generated = d_33_rolledGenerated_
                currentConstrainedOut = d_34_rolledCurrent_
                d_1_steps_ = (d_1_steps_) + (1)
                if ((parser).IsCompletePrefix(d_34_rolledCurrent_)) and ((d_1_steps_) < (maxSteps)):
                    d_35_closedGenerated2_: _dafny.Seq
                    d_36_closedInside2_: bool
                    d_37_closedCurrent2_: _dafny.Seq
                    out27_: _dafny.Seq
                    out28_: bool
                    out29_: _dafny.Seq
                    out27_, out28_, out29_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, d_34_rolledCurrent_)
                    d_35_closedGenerated2_ = out27_
                    d_36_closedInside2_ = out28_
                    d_37_closedCurrent2_ = out29_
                    generated = d_35_closedGenerated2_
                    insideConstrainedOut = d_36_closedInside2_
                    currentConstrainedOut = d_37_closedCurrent2_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    insideConstrainedOut = False
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

