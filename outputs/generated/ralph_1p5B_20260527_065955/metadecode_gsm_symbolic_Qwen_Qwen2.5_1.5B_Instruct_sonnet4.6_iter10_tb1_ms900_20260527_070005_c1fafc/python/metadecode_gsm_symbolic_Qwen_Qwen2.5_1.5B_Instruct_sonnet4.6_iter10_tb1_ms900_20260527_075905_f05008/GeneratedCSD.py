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
        (d_0_helpers_).AppendTaskGuidance(lm, ((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step using exact variable names from the problem. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Keep underscores: write n_1 not n1, k_2 not k2, big_fish not bigfish. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "At the end write ONLY the final answer expression inside << >> using Python arithmetic. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use +, -, *, /, //, %, int(). Example: <<int(n_1 * k_2 / d)>>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Do NOT use currency symbols or text inside << >>."))))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_unconstrainedSteps_: int
        d_2_unconstrainedSteps_ = 0
        d_3_spanTokenCount_: int
        d_3_spanTokenCount_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_2_unconstrainedSteps_) >= (200):
                            d_4_g2_: _dafny.Seq
                            d_5_i2_: bool
                            d_6_c2_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_g2_ = out0_
                            d_5_i2_ = out1_
                            d_6_c2_ = out2_
                            generated = d_4_g2_
                            insideConstrainedOut = d_5_i2_
                            currentConstrainedOut = d_6_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spanTokenCount_ = 0
                        elif True:
                            d_7_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_7_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_unconstrainedSteps_ = (d_2_unconstrainedSteps_) + (1)
                            if (d_7_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                                if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_3_spanTokenCount_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_8_g2_: _dafny.Seq
                        d_9_i2_: bool
                        d_10_c2_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_8_g2_ = out4_
                        d_9_i2_ = out5_
                        d_10_c2_ = out6_
                        generated = d_8_g2_
                        insideConstrainedOut = d_9_i2_
                        currentConstrainedOut = d_10_c2_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_spanTokenCount_ = 0
                        raise _dafny.Break("0")
                    elif True:
                        d_11_constrainedPrompt_: _dafny.Seq
                        d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_12_next_: _dafny.Seq
                        d_12_next_ = eosToken
                        if (d_3_spanTokenCount_) > (15):
                            d_13_penTokens_: _dafny.Seq
                            d_13_penTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "("))])
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('2e0'), d_13_penTokens_, _dafny.BigRational('5e0'), 8, eosToken)
                            d_12_next_ = out7_
                        elif True:
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_12_next_ = out8_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_12_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_14_g2_: _dafny.Seq
                            d_15_i2_: bool
                            d_16_c2_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                            d_14_g2_ = out9_
                            d_15_i2_ = out10_
                            d_16_c2_ = out11_
                            generated = d_14_g2_
                            insideConstrainedOut = d_15_i2_
                            currentConstrainedOut = d_16_c2_
                            d_3_spanTokenCount_ = (d_3_spanTokenCount_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

