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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Show your work, then write the final numeric answer inside << >> delimiters at the end.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeTokenCount_: int
        d_2_freeTokenCount_ = 0
        d_3_spanTokens_: int
        d_3_spanTokens_ = 0
        d_4_maxSpanTokens_: int
        d_4_maxSpanTokens_ = 25
        d_5_forceSpanThreshold_: int
        d_5_forceSpanThreshold_ = 80
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
                        d_15_wasConstrained_: bool
                        out7_: _dafny.Seq
                        out8_: bool
                        out7_, out8_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_14_next_ = out7_
                        d_15_wasConstrained_ = out8_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_14_next_) == (eosToken):
                            d_16_rolledGenerated_: _dafny.Seq
                            d_17_rolledCurrent_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: _dafny.Seq
                            out9_, out10_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_16_rolledGenerated_ = out9_
                            d_17_rolledCurrent_ = out10_
                            generated = d_16_rolledGenerated_
                            currentConstrainedOut = d_17_rolledCurrent_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                d_18_closedGenerated_: _dafny.Seq
                                d_19_closedInside_: bool
                                d_20_closedCurrent_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_18_closedGenerated_ = out11_
                                d_19_closedInside_ = out12_
                                d_20_closedCurrent_ = out13_
                                generated = d_18_closedGenerated_
                                insideConstrainedOut = d_19_closedInside_
                                currentConstrainedOut = d_20_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_3_spanTokens_ = 0
                                d_2_freeTokenCount_ = 0
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_3_spanTokens_ = 0
                        elif True:
                            d_21_valid_: bool
                            out14_: bool
                            out14_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_14_next_)
                            d_21_valid_ = out14_
                            if d_21_valid_:
                                d_22_appendedGenerated_: _dafny.Seq
                                d_23_appendedInside_: bool
                                d_24_appendedCurrent_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                d_22_appendedGenerated_ = out15_
                                d_23_appendedInside_ = out16_
                                d_24_appendedCurrent_ = out17_
                                generated = d_22_appendedGenerated_
                                insideConstrainedOut = d_23_appendedInside_
                                currentConstrainedOut = d_24_appendedCurrent_
                                d_3_spanTokens_ = (d_3_spanTokens_) + (1)
                                if (d_3_spanTokens_) >= (d_4_maxSpanTokens_):
                                    d_25_rolledGenerated_: _dafny.Seq
                                    d_26_rolledCurrent_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out18_, out19_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_25_rolledGenerated_ = out18_
                                    d_26_rolledCurrent_ = out19_
                                    generated = d_25_rolledGenerated_
                                    currentConstrainedOut = d_26_rolledCurrent_
                                    d_3_spanTokens_ = 0
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                        d_27_closedGenerated_: _dafny.Seq
                                        d_28_closedInside_: bool
                                        d_29_closedCurrent_: _dafny.Seq
                                        out20_: _dafny.Seq
                                        out21_: bool
                                        out22_: _dafny.Seq
                                        out20_, out21_, out22_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_27_closedGenerated_ = out20_
                                        d_28_closedInside_ = out21_
                                        d_29_closedCurrent_ = out22_
                                        generated = d_27_closedGenerated_
                                        insideConstrainedOut = d_28_closedInside_
                                        currentConstrainedOut = d_29_closedCurrent_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        d_2_freeTokenCount_ = 0
                                    elif True:
                                        insideConstrainedOut = False
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                        d_2_freeTokenCount_ = 0
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

