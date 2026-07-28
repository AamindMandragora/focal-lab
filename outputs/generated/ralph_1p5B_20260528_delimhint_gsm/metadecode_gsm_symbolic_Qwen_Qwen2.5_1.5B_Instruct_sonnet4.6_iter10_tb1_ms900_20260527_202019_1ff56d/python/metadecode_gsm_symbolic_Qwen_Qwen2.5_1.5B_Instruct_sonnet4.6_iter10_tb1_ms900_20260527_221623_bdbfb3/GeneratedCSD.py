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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Wrap every arithmetic expression and the final answer in << >> delimiters. Example: The total is <<3 * 5 = 15>>. The answer is <<15>>. Keep expressions short: use numbers and +, -, *, /, =, (, ) only.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanSteps_: int
        d_2_spanSteps_ = 0
        d_3_maxSpanSteps_: int
        d_3_maxSpanSteps_ = 20
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_remaining_: int
                        d_4_remaining_ = (maxSteps) - (d_1_steps_)
                        d_5_chunkBudget_: int
                        if (d_4_remaining_) < (40):
                            d_5_chunkBudget_ = d_4_remaining_
                        elif True:
                            d_5_chunkBudget_ = 40
                        if (d_5_chunkBudget_) == (0):
                            raise _dafny.Break("0")
                        d_6_g_: _dafny.Seq
                        d_7_stoppedOnOpen_: bool
                        d_8_stoppedOnEos_: bool
                        d_9_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_6_g_ = out0_
                        d_7_stoppedOnOpen_ = out1_
                        d_8_stoppedOnEos_ = out2_
                        d_9_stepsUsed_ = out3_
                        generated = d_6_g_
                        d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                        if d_8_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_7_stoppedOnOpen_:
                            d_10_g2_: _dafny.Seq
                            d_11_i2_: bool
                            d_12_c2_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_10_g2_ = out4_
                            d_11_i2_ = out5_
                            d_12_c2_ = out6_
                            generated = d_10_g2_
                            insideConstrainedOut = d_11_i2_
                            currentConstrainedOut = d_12_c2_
                            d_2_spanSteps_ = 0
                        elif True:
                            if (d_1_steps_) < (maxSteps):
                                d_13_g2_: _dafny.Seq
                                d_14_i2_: bool
                                d_15_c2_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_13_g2_ = out7_
                                d_14_i2_ = out8_
                                d_15_c2_ = out9_
                                generated = d_13_g2_
                                insideConstrainedOut = d_14_i2_
                                currentConstrainedOut = d_15_c2_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_spanSteps_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_16_g_: _dafny.Seq
                        d_17_i_: bool
                        d_18_c_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_16_g_ = out10_
                        d_17_i_ = out11_
                        d_18_c_ = out12_
                        generated = d_16_g_
                        insideConstrainedOut = d_17_i_
                        currentConstrainedOut = d_18_c_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanSteps_ = 0
                    elif (d_2_spanSteps_) >= (d_3_maxSpanSteps_):
                        d_19_rolledG_: _dafny.Seq
                        d_20_rolledC_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: _dafny.Seq
                        out13_, out14_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_19_rolledG_ = out13_
                        d_20_rolledC_ = out14_
                        generated = d_19_rolledG_
                        currentConstrainedOut = d_20_rolledC_
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_21_g_: _dafny.Seq
                            d_22_i_: bool
                            d_23_c_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_21_g_ = out15_
                            d_22_i_ = out16_
                            d_23_c_ = out17_
                            generated = d_21_g_
                            insideConstrainedOut = d_22_i_
                            currentConstrainedOut = d_23_c_
                        elif True:
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanSteps_ = 0
                    elif True:
                        d_24_constrainedPrompt_: _dafny.Seq
                        d_24_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_25_next_: _dafny.Seq
                        out18_: _dafny.Seq
                        out18_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                        d_25_next_ = out18_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                        if (d_25_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_26_g_: _dafny.Seq
                            d_27_i_: bool
                            d_28_c_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: _dafny.Seq
                            out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next_)
                            d_26_g_ = out19_
                            d_27_i_ = out20_
                            d_28_c_ = out21_
                            generated = d_26_g_
                            insideConstrainedOut = d_27_i_
                            currentConstrainedOut = d_28_c_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

