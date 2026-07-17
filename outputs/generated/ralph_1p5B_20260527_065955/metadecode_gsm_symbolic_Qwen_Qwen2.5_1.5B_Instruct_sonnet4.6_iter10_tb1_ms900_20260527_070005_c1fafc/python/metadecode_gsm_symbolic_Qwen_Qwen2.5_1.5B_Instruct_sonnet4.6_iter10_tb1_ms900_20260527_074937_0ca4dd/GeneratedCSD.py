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
        (d_0_helpers_).AppendTaskGuidance(lm, ((((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem carefully using only the variable names given in the problem. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Show your reasoning, then write each key calculation inside << >>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "The very last << >> must contain ONLY the final answer expression. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use exact variable names from the problem (like n1, n2, t, price, etc.). ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Valid operators: +, -, *, /, //, %, int(). ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Example final answer: <<(n1 + n2) * 7>> or <<int(a * b / c)>>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Do NOT use undefined names or currency symbols inside << >>."))))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanLength_: int
        d_2_spanLength_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_remaining_: int
                        d_3_remaining_ = (maxSteps) - (d_1_steps_)
                        d_4_chunkBudget_: int
                        if (d_3_remaining_) > (30):
                            d_4_chunkBudget_ = 30
                        elif True:
                            d_4_chunkBudget_ = d_3_remaining_
                        if (d_4_chunkBudget_) == (0):
                            raise _dafny.Break("0")
                        d_5_generatedOut_: _dafny.Seq
                        d_6_stoppedOnOpenSpan_: bool
                        d_7_stoppedOnEos_: bool
                        d_8_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_5_generatedOut_ = out0_
                        d_6_stoppedOnOpenSpan_ = out1_
                        d_7_stoppedOnEos_ = out2_
                        d_8_stepsUsed_ = out3_
                        generated = d_5_generatedOut_
                        d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                        if d_7_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_6_stoppedOnOpenSpan_:
                            d_9_g2_: _dafny.Seq
                            d_10_i2_: bool
                            d_11_c2_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_9_g2_ = out4_
                            d_10_i2_ = out5_
                            d_11_c2_ = out6_
                            generated = d_9_g2_
                            insideConstrainedOut = d_10_i2_
                            currentConstrainedOut = d_11_c2_
                            d_2_spanLength_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_12_g2_: _dafny.Seq
                        d_13_i2_: bool
                        d_14_c2_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_12_g2_ = out7_
                        d_13_i2_ = out8_
                        d_14_c2_ = out9_
                        generated = d_12_g2_
                        insideConstrainedOut = d_13_i2_
                        currentConstrainedOut = d_14_c2_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanLength_ = 0
                    elif True:
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_16_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('3e0'), eosToken)
                        d_16_next_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_16_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_17_g2_: _dafny.Seq
                            d_18_i2_: bool
                            d_19_c2_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                            d_17_g2_ = out11_
                            d_18_i2_ = out12_
                            d_19_c2_ = out13_
                            generated = d_17_g2_
                            insideConstrainedOut = d_18_i2_
                            currentConstrainedOut = d_19_c2_
                            d_2_spanLength_ = (d_2_spanLength_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

