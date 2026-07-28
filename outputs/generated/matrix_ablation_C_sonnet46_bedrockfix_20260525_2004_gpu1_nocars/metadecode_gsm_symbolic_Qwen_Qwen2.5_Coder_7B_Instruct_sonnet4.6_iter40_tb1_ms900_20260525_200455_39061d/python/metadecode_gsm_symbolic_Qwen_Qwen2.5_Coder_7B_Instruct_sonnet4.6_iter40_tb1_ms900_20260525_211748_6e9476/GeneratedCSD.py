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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Use the variable names from the problem. At the end, write ONLY the final answer expression inside << >> delimiters using the problem variables. Example: <<n1 * c1 + n2 * c2>>. Keep the expression simple and correct. Do not repeat the same expression multiple times.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_maxConstrainedLen_: int
        d_2_maxConstrainedLen_ = 40
        d_3_hasOpenedSpan_: bool
        d_3_hasOpenedSpan_ = insideConstrained
        d_4_forcedOpen_: bool
        d_4_forcedOpen_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_4_forcedOpen_:
                            d_5_openedGenerated_: _dafny.Seq
                            d_6_openedInside_: bool
                            d_7_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_openedGenerated_ = out0_
                            d_6_openedInside_ = out1_
                            d_7_openedCurrent_ = out2_
                            generated = d_5_openedGenerated_
                            insideConstrainedOut = d_6_openedInside_
                            currentConstrainedOut = d_7_openedCurrent_
                            d_3_hasOpenedSpan_ = True
                            d_4_forcedOpen_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_chunkBudget_: int
                            if ((maxSteps) - (d_1_steps_)) > (500):
                                d_8_chunkBudget_ = 500
                            elif True:
                                d_8_chunkBudget_ = (maxSteps) - (d_1_steps_)
                            d_9_chunkedG_: _dafny.Seq
                            d_10_stoppedOpen_: bool
                            d_11_stoppedEos_: bool
                            d_12_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_9_chunkedG_ = out3_
                            d_10_stoppedOpen_ = out4_
                            d_11_stoppedEos_ = out5_
                            d_12_stepsUsed_ = out6_
                            generated = d_9_chunkedG_
                            d_1_steps_ = (d_1_steps_) + (d_12_stepsUsed_)
                            if d_10_stoppedOpen_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_3_hasOpenedSpan_ = True
                                d_4_forcedOpen_ = False
                            elif d_11_stoppedEos_:
                                if (not(d_3_hasOpenedSpan_)) and ((d_1_steps_) < (maxSteps)):
                                    d_4_forcedOpen_ = True
                                elif True:
                                    raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_13_closedGenerated_: _dafny.Seq
                        d_14_closedInside_: bool
                        d_15_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_13_closedGenerated_ = out7_
                        d_14_closedInside_ = out8_
                        d_15_closedCurrent_ = out9_
                        generated = d_13_closedGenerated_
                        insideConstrainedOut = d_14_closedInside_
                        currentConstrainedOut = d_15_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_2_maxConstrainedLen_):
                        d_16_rolledGenerated_: _dafny.Seq
                        d_17_rolledCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out10_, out11_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_16_rolledGenerated_ = out10_
                        d_17_rolledCurrent_ = out11_
                        generated = d_16_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_17_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_next_: _dafny.Seq
                        out12_: _dafny.Seq
                        out12_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('4e0'), eosToken)
                        d_19_next_ = out12_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_19_next_) == (eosToken):
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                pass
                            elif True:
                                d_20_rolledGenerated_: _dafny.Seq
                                d_21_rolledCurrent_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: _dafny.Seq
                                out13_, out14_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_20_rolledGenerated_ = out13_
                                d_21_rolledCurrent_ = out14_
                                generated = d_20_rolledGenerated_
                                currentConstrainedOut = d_21_rolledCurrent_
                                if ((parser).IsCompletePrefix(d_21_rolledCurrent_)) and ((d_1_steps_) < (maxSteps)):
                                    d_22_closedGenerated_: _dafny.Seq
                                    d_23_closedInside_: bool
                                    d_24_closedCurrent_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, d_21_rolledCurrent_)
                                    d_22_closedGenerated_ = out15_
                                    d_23_closedInside_ = out16_
                                    d_24_closedCurrent_ = out17_
                                    generated = d_22_closedGenerated_
                                    insideConstrainedOut = d_23_closedInside_
                                    currentConstrainedOut = d_24_closedCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                        elif True:
                            d_25_appendedGenerated_: _dafny.Seq
                            d_26_appendedInside_: bool
                            d_27_appendedCurrent_: _dafny.Seq
                            out18_: _dafny.Seq
                            out19_: bool
                            out20_: _dafny.Seq
                            out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                            d_25_appendedGenerated_ = out18_
                            d_26_appendedInside_ = out19_
                            d_27_appendedCurrent_ = out20_
                            generated = d_25_appendedGenerated_
                            insideConstrainedOut = d_26_appendedInside_
                            currentConstrainedOut = d_27_appendedCurrent_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_28_closedGenerated_: _dafny.Seq
                d_29_closedInside_: bool
                d_30_closedCurrent_: _dafny.Seq
                out21_: _dafny.Seq
                out22_: bool
                out23_: _dafny.Seq
                out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_28_closedGenerated_ = out21_
                d_29_closedInside_ = out22_
                d_30_closedCurrent_ = out23_
                generated = d_28_closedGenerated_
                insideConstrainedOut = d_29_closedInside_
                currentConstrainedOut = d_30_closedCurrent_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_31_rolledGenerated_: _dafny.Seq
                d_32_rolledCurrent_: _dafny.Seq
                out24_: _dafny.Seq
                out25_: _dafny.Seq
                out24_, out25_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                d_31_rolledGenerated_ = out24_
                d_32_rolledCurrent_ = out25_
                generated = d_31_rolledGenerated_
                currentConstrainedOut = d_32_rolledCurrent_
                d_1_steps_ = (d_1_steps_) + (1)
                if ((parser).IsCompletePrefix(d_32_rolledCurrent_)) and ((d_1_steps_) < (maxSteps)):
                    d_33_closedGenerated2_: _dafny.Seq
                    d_34_closedInside2_: bool
                    d_35_closedCurrent2_: _dafny.Seq
                    out26_: _dafny.Seq
                    out27_: bool
                    out28_: _dafny.Seq
                    out26_, out27_, out28_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, d_32_rolledCurrent_)
                    d_33_closedGenerated2_ = out26_
                    d_34_closedInside2_ = out27_
                    d_35_closedCurrent2_ = out28_
                    generated = d_33_closedGenerated2_
                    insideConstrainedOut = d_34_closedInside2_
                    currentConstrainedOut = d_35_closedCurrent2_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    insideConstrainedOut = False
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

