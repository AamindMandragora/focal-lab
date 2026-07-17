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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Show each intermediate calculation inside << >> delimiters, for example <<3+4=7>>. End your answer with the final numeric result also inside << >> delimiters, like <<42>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeChunkSize_: int
        d_2_freeChunkSize_ = 12
        d_3_freeTokensInChunk_: int
        d_3_freeTokensInChunk_ = 0
        d_4_justClosed_: bool
        d_4_justClosed_ = False
        d_5_minFreeAfterClose_: int
        d_5_minFreeAfterClose_ = 3
        d_6_freeAfterClose_: int
        d_6_freeAfterClose_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (((d_3_freeTokensInChunk_) >= (d_2_freeChunkSize_)) and ((d_6_freeAfterClose_) >= (d_5_minFreeAfterClose_))) and (((d_1_steps_) + (2)) <= (maxSteps)):
                            d_7_g2_: _dafny.Seq
                            d_8_ins2_: bool
                            d_9_cur2_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_7_g2_ = out0_
                            d_8_ins2_ = out1_
                            d_9_cur2_ = out2_
                            generated = d_7_g2_
                            insideConstrainedOut = d_8_ins2_
                            currentConstrainedOut = d_9_cur2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_freeTokensInChunk_ = 0
                            d_6_freeAfterClose_ = 0
                            d_4_justClosed_ = False
                        elif True:
                            d_10_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_10_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_10_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                                d_3_freeTokensInChunk_ = (d_3_freeTokensInChunk_) + (1)
                                if d_4_justClosed_:
                                    d_6_freeAfterClose_ = (d_6_freeAfterClose_) + (1)
                                if (d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_11_g2_: _dafny.Seq
                                    d_12_ins2_: bool
                                    d_13_cur2_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_11_g2_ = out4_
                                    d_12_ins2_ = out5_
                                    d_13_cur2_ = out6_
                                    generated = d_11_g2_
                                    insideConstrainedOut = d_12_ins2_
                                    currentConstrainedOut = d_13_cur2_
                                    d_3_freeTokensInChunk_ = 0
                                    d_6_freeAfterClose_ = 0
                                    d_4_justClosed_ = False
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_14_closedGenerated_: _dafny.Seq
                        d_15_closedInside_: bool
                        d_16_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_14_closedGenerated_ = out7_
                        d_15_closedInside_ = out8_
                        d_16_closedCurrent_ = out9_
                        generated = d_14_closedGenerated_
                        insideConstrainedOut = d_15_closedInside_
                        currentConstrainedOut = d_16_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_4_justClosed_ = True
                        d_6_freeAfterClose_ = 0
                    elif True:
                        d_17_constrainedPrompt_: _dafny.Seq
                        d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_18_next_: _dafny.Seq
                        d_19_wasConstrained_: bool
                        out10_: _dafny.Seq
                        out11_: bool
                        out10_, out11_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_18_next_ = out10_
                        d_19_wasConstrained_ = out11_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_18_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_20_appendedGenerated_: _dafny.Seq
                            d_21_appendedInside_: bool
                            d_22_appendedCurrent_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                            d_20_appendedGenerated_ = out12_
                            d_21_appendedInside_ = out13_
                            d_22_appendedCurrent_ = out14_
                            generated = d_20_appendedGenerated_
                            insideConstrainedOut = d_21_appendedInside_
                            currentConstrainedOut = d_22_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

