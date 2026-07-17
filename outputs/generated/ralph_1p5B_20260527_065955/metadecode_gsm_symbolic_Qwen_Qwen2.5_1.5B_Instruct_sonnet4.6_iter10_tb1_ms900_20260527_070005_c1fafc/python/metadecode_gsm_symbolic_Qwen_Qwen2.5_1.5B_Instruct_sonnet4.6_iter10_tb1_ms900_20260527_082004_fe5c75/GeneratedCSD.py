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
        (d_0_helpers_).AppendTaskGuidance(lm, ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Wrap every intermediate expression and the final answer in << >>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use Python arithmetic: +, -, *, /, //, %. Keep each <<expression>> short."))))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanTokenCount_: int
        d_2_spanTokenCount_ = 0
        d_3_unconstrainedCount_: int
        d_3_unconstrainedCount_ = 0
        d_4_spansOpened_: int
        d_4_spansOpened_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_3_unconstrainedCount_) >= (8):
                            d_5_g2_: _dafny.Seq
                            d_6_i2_: bool
                            d_7_c2_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_g2_ = out0_
                            d_6_i2_ = out1_
                            d_7_c2_ = out2_
                            generated = d_5_g2_
                            insideConstrainedOut = d_6_i2_
                            currentConstrainedOut = d_7_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_spanTokenCount_ = 0
                            d_4_spansOpened_ = (d_4_spansOpened_) + (1)
                        elif (d_4_spansOpened_) >= (2):
                            d_8_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_spanTokenCount_ = 0
                                    d_4_spansOpened_ = (d_4_spansOpened_) + (1)
                                elif True:
                                    d_3_unconstrainedCount_ = (d_3_unconstrainedCount_) + (1)
                        elif True:
                            d_9_next_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_9_next_ = out4_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                if (d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_spanTokenCount_ = 0
                                    d_4_spansOpened_ = (d_4_spansOpened_) + (1)
                                elif True:
                                    d_3_unconstrainedCount_ = (d_3_unconstrainedCount_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_10_g2_: _dafny.Seq
                        d_11_i2_: bool
                        d_12_c2_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_10_g2_ = out5_
                        d_11_i2_ = out6_
                        d_12_c2_ = out7_
                        generated = d_10_g2_
                        insideConstrainedOut = d_11_i2_
                        currentConstrainedOut = d_12_c2_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanTokenCount_ = 0
                        d_3_unconstrainedCount_ = 0
                    elif (d_2_spanTokenCount_) >= (28):
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_13_constrainedPrompt_: _dafny.Seq
                        d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_14_next_: _dafny.Seq
                        out8_: _dafny.Seq
                        out8_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_14_next_ = out8_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_14_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_15_g2_: _dafny.Seq
                            d_16_i2_: bool
                            d_17_c2_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                            d_15_g2_ = out9_
                            d_16_i2_ = out10_
                            d_17_c2_ = out11_
                            generated = d_15_g2_
                            insideConstrainedOut = d_16_i2_
                            currentConstrainedOut = d_17_c2_
                            d_2_spanTokenCount_ = (d_2_spanTokenCount_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

