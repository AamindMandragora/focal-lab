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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the word problem step by step. Put intermediate calculations and the final answer inside visible << >> spans. Inside each span, write only a compact arithmetic expression or number, with no words or extra delimiters."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_done_: bool
        d_3_done_ = False
        if (maxSteps) > (0):
            if not(insideConstrainedOut):
                d_4_openedGenerated_: _dafny.Seq
                d_5_openedInside_: bool
                d_6_openedCurrent_: _dafny.Seq
                out0_: _dafny.Seq
                out1_: bool
                out2_: _dafny.Seq
                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_4_openedGenerated_ = out0_
                d_5_openedInside_ = out1_
                d_6_openedCurrent_ = out2_
                generated = d_4_openedGenerated_
                insideConstrainedOut = d_5_openedInside_
                currentConstrainedOut = d_6_openedCurrent_
                d_2_steps_ = 1
            elif (parser).IsCompletePrefix(currentConstrainedOut):
                d_7_initiallyClosedGenerated_: _dafny.Seq
                d_8_initiallyClosedInside_: bool
                d_9_initiallyClosedCurrent_: _dafny.Seq
                out3_: _dafny.Seq
                out4_: bool
                out5_: _dafny.Seq
                out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_7_initiallyClosedGenerated_ = out3_
                d_8_initiallyClosedInside_ = out4_
                d_9_initiallyClosedCurrent_ = out5_
                generated = d_7_initiallyClosedGenerated_
                insideConstrainedOut = d_8_initiallyClosedInside_
                currentConstrainedOut = d_9_initiallyClosedCurrent_
                d_2_steps_ = 1
                d_3_done_ = True
            elif True:
                d_10_initialConstrainedPrompt_: _dafny.Seq
                d_10_initialConstrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_11_initialNext_: _dafny.Seq
                out6_: _dafny.Seq
                out6_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_10_initialConstrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                d_11_initialNext_ = out6_
                d_2_steps_ = 1
                if (d_11_initialNext_) != (eosToken):
                    d_12_initialValid_: bool
                    out7_: bool
                    out7_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_11_initialNext_)
                    d_12_initialValid_ = out7_
                    if d_12_initialValid_:
                        d_13_initiallyAppendedGenerated_: _dafny.Seq
                        d_14_initiallyAppendedInside_: bool
                        d_15_initiallyAppendedCurrent_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_initialNext_)
                        d_13_initiallyAppendedGenerated_ = out8_
                        d_14_initiallyAppendedInside_ = out9_
                        d_15_initiallyAppendedCurrent_ = out10_
                        generated = d_13_initiallyAppendedGenerated_
                        insideConstrainedOut = d_14_initiallyAppendedInside_
                        currentConstrainedOut = d_15_initiallyAppendedCurrent_
            while ((d_2_steps_) < (maxSteps)) and (not(d_3_done_)):
                if not(insideConstrainedOut):
                    d_16_loopOpenedGenerated_: _dafny.Seq
                    d_17_loopOpenedInside_: bool
                    d_18_loopOpenedCurrent_: _dafny.Seq
                    out11_: _dafny.Seq
                    out12_: bool
                    out13_: _dafny.Seq
                    out11_, out12_, out13_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_16_loopOpenedGenerated_ = out11_
                    d_17_loopOpenedInside_ = out12_
                    d_18_loopOpenedCurrent_ = out13_
                    generated = d_16_loopOpenedGenerated_
                    insideConstrainedOut = d_17_loopOpenedInside_
                    currentConstrainedOut = d_18_loopOpenedCurrent_
                    d_2_steps_ = (d_2_steps_) + (1)
                elif (parser).IsCompletePrefix(currentConstrainedOut):
                    d_19_closedGenerated_: _dafny.Seq
                    d_20_closedInside_: bool
                    d_21_closedCurrent_: _dafny.Seq
                    out14_: _dafny.Seq
                    out15_: bool
                    out16_: _dafny.Seq
                    out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_19_closedGenerated_ = out14_
                    d_20_closedInside_ = out15_
                    d_21_closedCurrent_ = out16_
                    generated = d_19_closedGenerated_
                    insideConstrainedOut = d_20_closedInside_
                    currentConstrainedOut = d_21_closedCurrent_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_3_done_ = True
                elif True:
                    d_22_constrainedPrompt_: _dafny.Seq
                    d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_23_nextConstrained_: _dafny.Seq
                    out17_: _dafny.Seq
                    out17_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                    d_23_nextConstrained_ = out17_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_23_nextConstrained_) != (eosToken):
                        d_24_validNext_: bool
                        out18_: bool
                        out18_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_23_nextConstrained_)
                        d_24_validNext_ = out18_
                        if d_24_validNext_:
                            d_25_appendedGenerated_: _dafny.Seq
                            d_26_appendedInside_: bool
                            d_27_appendedCurrent_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: _dafny.Seq
                            out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_nextConstrained_)
                            d_25_appendedGenerated_ = out19_
                            d_26_appendedInside_ = out20_
                            d_27_appendedCurrent_ = out21_
                            generated = d_25_appendedGenerated_
                            insideConstrainedOut = d_26_appendedInside_
                            currentConstrainedOut = d_27_appendedCurrent_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                d_28_justClosedGenerated_: _dafny.Seq
                                d_29_justClosedInside_: bool
                                d_30_justClosedCurrent_: _dafny.Seq
                                out22_: _dafny.Seq
                                out23_: bool
                                out24_: _dafny.Seq
                                out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_28_justClosedGenerated_ = out22_
                                d_29_justClosedInside_ = out23_
                                d_30_justClosedCurrent_ = out24_
                                generated = d_28_justClosedGenerated_
                                insideConstrainedOut = d_29_justClosedInside_
                                currentConstrainedOut = d_30_justClosedCurrent_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_3_done_ = True
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

