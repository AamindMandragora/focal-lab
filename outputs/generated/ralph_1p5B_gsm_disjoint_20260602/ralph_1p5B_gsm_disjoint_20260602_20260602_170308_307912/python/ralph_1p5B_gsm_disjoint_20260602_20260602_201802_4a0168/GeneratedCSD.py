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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Write your reasoning, then put the final numeric answer inside << >> delimiters, e.g. <<42>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeTokenCount_: int
        d_2_freeTokenCount_ = 0
        d_3_spanTokens_: int
        d_3_spanTokens_ = 0
        d_4_maxSpanTokens_: int
        d_4_maxSpanTokens_ = 8
        d_5_forceSpanThreshold_: int
        d_5_forceSpanThreshold_ = 45
        d_6_closedSpanCount_: int
        d_6_closedSpanCount_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_2_freeTokenCount_) >= (d_5_forceSpanThreshold_)) and (((d_1_steps_) + (3)) <= (maxSteps)):
                            d_7_openGenerated_: _dafny.Seq
                            d_8_openInside_: bool
                            d_9_openCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_7_openGenerated_ = out0_
                            d_8_openInside_ = out1_
                            d_9_openCurrent_ = out2_
                            generated = d_7_openGenerated_
                            insideConstrainedOut = d_8_openInside_
                            currentConstrainedOut = d_9_openCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spanTokens_ = 0
                            d_2_freeTokenCount_ = 0
                        elif True:
                            d_10_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_10_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_10_next_) == (eosToken):
                                if ((d_6_closedSpanCount_) == (0)) and (((d_1_steps_) + (3)) <= (maxSteps)):
                                    d_11_openGenerated_: _dafny.Seq
                                    d_12_openInside_: bool
                                    d_13_openCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_11_openGenerated_ = out4_
                                    d_12_openInside_ = out5_
                                    d_13_openCurrent_ = out6_
                                    generated = d_11_openGenerated_
                                    insideConstrainedOut = d_12_openInside_
                                    currentConstrainedOut = d_13_openCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_3_spanTokens_ = 0
                                    d_2_freeTokenCount_ = 0
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                                if (d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_3_spanTokens_ = 0
                                    d_2_freeTokenCount_ = 0
                                elif True:
                                    d_2_freeTokenCount_ = (d_2_freeTokenCount_) + (1)
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
                        d_3_spanTokens_ = 0
                        d_2_freeTokenCount_ = 0
                        d_6_closedSpanCount_ = (d_6_closedSpanCount_) + (1)
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
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_3_spanTokens_ = 0
                        elif True:
                            d_20_valid_: bool
                            out12_: bool
                            out12_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_18_next_)
                            d_20_valid_ = out12_
                            if d_20_valid_:
                                d_21_appendedGenerated_: _dafny.Seq
                                d_22_appendedInside_: bool
                                d_23_appendedCurrent_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                                d_21_appendedGenerated_ = out13_
                                d_22_appendedInside_ = out14_
                                d_23_appendedCurrent_ = out15_
                                generated = d_21_appendedGenerated_
                                insideConstrainedOut = d_22_appendedInside_
                                currentConstrainedOut = d_23_appendedCurrent_
                                d_3_spanTokens_ = (d_3_spanTokens_) + (1)
                                if (d_3_spanTokens_) >= (d_4_maxSpanTokens_):
                                    d_24_rolledGenerated_: _dafny.Seq
                                    d_25_rolledCurrent_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out16_, out17_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_24_rolledGenerated_ = out16_
                                    d_25_rolledCurrent_ = out17_
                                    generated = d_24_rolledGenerated_
                                    currentConstrainedOut = d_25_rolledCurrent_
                                    d_3_spanTokens_ = 0
                                    if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                        insideConstrainedOut = False
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                        d_2_freeTokenCount_ = 0
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

