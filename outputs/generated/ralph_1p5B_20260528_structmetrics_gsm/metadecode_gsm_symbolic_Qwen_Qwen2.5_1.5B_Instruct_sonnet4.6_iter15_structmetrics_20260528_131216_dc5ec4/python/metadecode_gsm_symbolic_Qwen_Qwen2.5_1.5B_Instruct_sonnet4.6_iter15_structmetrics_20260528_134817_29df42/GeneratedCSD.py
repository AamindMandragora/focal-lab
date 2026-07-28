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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math problem step by step. After each sentence of reasoning, put the arithmetic computation inside << >> delimiters, like: <<3 * 4 = 12>>. Every intermediate calculation must appear in << >>. The final numeric answer must also be inside << >> delimiters.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeStepsSinceLastSpan_: int
        d_2_freeStepsSinceLastSpan_ = 0
        d_3_forceSpanAfter_: int
        d_3_forceSpanAfter_ = 45
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_2_freeStepsSinceLastSpan_) >= (d_3_forceSpanAfter_):
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
                            d_2_freeStepsSinceLastSpan_ = 0
                        elif True:
                            d_7_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_7_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                                if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_freeStepsSinceLastSpan_ = 0
                                elif True:
                                    d_2_freeStepsSinceLastSpan_ = (d_2_freeStepsSinceLastSpan_) + (1)
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
                        d_2_freeStepsSinceLastSpan_ = 0
                    elif True:
                        d_11_constrainedPrompt_: _dafny.Seq
                        d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_12_next_: _dafny.Seq
                        d_13_wasConstrained_: bool
                        out7_: _dafny.Seq
                        out8_: bool
                        out7_, out8_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_12_next_ = out7_
                        d_13_wasConstrained_ = out8_
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
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

