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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Show your work, then write the final answer inside << >> delimiters, e.g. <<42>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeTokenCount_: int
        d_2_freeTokenCount_ = 0
        d_3_spanTokens_: int
        d_3_spanTokens_ = 0
        d_4_maxSpanTokens_: int
        d_4_maxSpanTokens_ = 35
        d_5_forceSpanThreshold_: int
        d_5_forceSpanThreshold_ = 50
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_2_freeTokenCount_) >= (d_5_forceSpanThreshold_)) and (((d_1_steps_) + (3)) <= (maxSteps)):
                            d_6_openGenerated_: _dafny.Seq
                            d_7_openInside_: bool
                            d_8_openCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_openGenerated_ = out0_
                            d_7_openInside_ = out1_
                            d_8_openCurrent_ = out2_
                            generated = d_6_openGenerated_
                            insideConstrainedOut = d_7_openInside_
                            currentConstrainedOut = d_8_openCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spanTokens_ = 0
                            d_2_freeTokenCount_ = 0
                        elif True:
                            d_9_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_9_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                if (d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_3_spanTokens_ = 0
                                    d_2_freeTokenCount_ = 0
                                elif True:
                                    d_2_freeTokenCount_ = (d_2_freeTokenCount_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_10_closedGenerated_: _dafny.Seq
                        d_11_closedInside_: bool
                        d_12_closedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_10_closedGenerated_ = out4_
                        d_11_closedInside_ = out5_
                        d_12_closedCurrent_ = out6_
                        generated = d_10_closedGenerated_
                        insideConstrainedOut = d_11_closedInside_
                        currentConstrainedOut = d_12_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_spanTokens_ = 0
                        d_2_freeTokenCount_ = 0
                    elif True:
                        d_13_constrainedPrompt_: _dafny.Seq
                        d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_14_next_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_14_next_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_14_next_) == (eosToken):
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_3_spanTokens_ = 0
                        elif True:
                            d_15_valid_: bool
                            out8_: bool
                            out8_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_14_next_)
                            d_15_valid_ = out8_
                            if d_15_valid_:
                                d_16_appendedGenerated_: _dafny.Seq
                                d_17_appendedInside_: bool
                                d_18_appendedCurrent_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                d_16_appendedGenerated_ = out9_
                                d_17_appendedInside_ = out10_
                                d_18_appendedCurrent_ = out11_
                                generated = d_16_appendedGenerated_
                                insideConstrainedOut = d_17_appendedInside_
                                currentConstrainedOut = d_18_appendedCurrent_
                                d_3_spanTokens_ = (d_3_spanTokens_) + (1)
                                if (d_3_spanTokens_) >= (d_4_maxSpanTokens_):
                                    d_19_rolledGenerated_: _dafny.Seq
                                    d_20_rolledCurrent_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out12_, out13_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_19_rolledGenerated_ = out12_
                                    d_20_rolledCurrent_ = out13_
                                    generated = d_19_rolledGenerated_
                                    currentConstrainedOut = d_20_rolledCurrent_
                                    d_3_spanTokens_ = 0
                                    if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                        insideConstrainedOut = False
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                        d_2_freeTokenCount_ = 0
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

