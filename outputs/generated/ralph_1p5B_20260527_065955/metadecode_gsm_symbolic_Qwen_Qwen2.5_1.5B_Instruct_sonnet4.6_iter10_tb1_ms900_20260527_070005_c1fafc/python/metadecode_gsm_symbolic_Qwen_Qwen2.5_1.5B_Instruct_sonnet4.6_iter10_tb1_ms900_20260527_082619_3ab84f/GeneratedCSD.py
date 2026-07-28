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
        (d_0_helpers_).AppendTaskGuidance(lm, (((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "For EACH intermediate calculation, write << expression >>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use the exact variable names from the problem statement. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Keep each expression under 6 tokens. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Put the FINAL numeric answer inside << >> at the end. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Do not repeat operations. Do not extend expressions unnecessarily."))))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanTokenCount_: int
        d_2_spanTokenCount_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_3_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                            if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_spanTokenCount_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_4_g2_: _dafny.Seq
                        d_5_i2_: bool
                        d_6_c2_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_4_g2_ = out1_
                        d_5_i2_ = out2_
                        d_6_c2_ = out3_
                        generated = d_4_g2_
                        insideConstrainedOut = d_5_i2_
                        currentConstrainedOut = d_6_c2_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanTokenCount_ = 0
                    elif True:
                        d_7_constrainedPrompt_: _dafny.Seq
                        d_7_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_8_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if (d_2_spanTokenCount_) > (8):
                            d_9_penTokens_: _dafny.Seq
                            d_9_penTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "//")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "%")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9"))])
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('2e0'), d_9_penTokens_, _dafny.BigRational('8e0'), 4, eosToken)
                            d_8_next_ = out4_
                        elif True:
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('3e0'), eosToken)
                            d_8_next_ = out5_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_8_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_10_g2_: _dafny.Seq
                            d_11_i2_: bool
                            d_12_c2_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_8_next_)
                            d_10_g2_ = out6_
                            d_11_i2_ = out7_
                            d_12_c2_ = out8_
                            generated = d_10_g2_
                            insideConstrainedOut = d_11_i2_
                            currentConstrainedOut = d_12_c2_
                            d_2_spanTokenCount_ = (d_2_spanTokenCount_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

