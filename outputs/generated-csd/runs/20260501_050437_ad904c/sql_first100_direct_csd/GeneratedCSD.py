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
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        d_3_fromKeyword_: _dafny.Seq
        d_3_fromKeyword_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))
        d_4_selectKeyword_: _dafny.Seq
        d_4_selectKeyword_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT"))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_5_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_5_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                            if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_6_complete_: bool
                        d_6_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_6_complete_:
                            d_7_closedGenerated_: _dafny.Seq
                            d_8_closedInside_: bool
                            d_9_closedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_7_closedGenerated_ = out1_
                            d_8_closedInside_ = out2_
                            d_9_closedCurrent_ = out3_
                            generated = d_7_closedGenerated_
                            insideConstrainedOut = d_8_closedInside_
                            currentConstrainedOut = d_9_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_10_stablePrefix_: _dafny.Seq
                            d_10_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_11_constrainedPrompt_: _dafny.Seq
                            d_11_constrainedPrompt_ = (prompt) + (d_10_stablePrefix_)
                            d_12_validCount_: int
                            out4_: int
                            out4_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_12_validCount_ = out4_
                            if (d_12_validCount_) <= (d_2_narrowThreshold_):
                                d_13_next_: _dafny.Seq
                                out5_: _dafny.Seq
                                out5_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                                d_13_next_ = out5_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_13_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_14_appendedGenerated_: _dafny.Seq
                                    d_15_appendedInside_: bool
                                    d_16_appendedCurrent_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out7_: bool
                                    out8_: _dafny.Seq
                                    out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                                    d_14_appendedGenerated_ = out6_
                                    d_15_appendedInside_ = out7_
                                    d_16_appendedCurrent_ = out8_
                                    generated = d_14_appendedGenerated_
                                    insideConstrainedOut = d_15_appendedInside_
                                    currentConstrainedOut = d_16_appendedCurrent_
                            elif True:
                                d_17_fromContext_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_3_fromKeyword_)
                                d_17_fromContext_ = out9_
                                d_18_selectContext_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_4_selectKeyword_)
                                d_18_selectContext_ = out10_
                                d_19_flatGroups_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                d_19_flatGroups_ = out11_
                                (lm).GenerateLogits((d_11_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(d_19_flatGroups_)) > (0):
                                    (d_0_helpers_).BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'))
                                if (len(d_17_fromContext_)) > (0):
                                    d_20_anyFromValid_: bool
                                    out12_: bool
                                    out12_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_17_fromContext_)
                                    d_20_anyFromValid_ = out12_
                                    if d_20_anyFromValid_:
                                        (d_0_helpers_).BoostTokenLogits(lm, d_17_fromContext_, _dafny.BigRational('6e0'))
                                if (len(d_18_selectContext_)) > (0):
                                    d_21_anySelectValid_: bool
                                    out13_: bool
                                    out13_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_18_selectContext_)
                                    d_21_anySelectValid_ = out13_
                                    if d_21_anySelectValid_:
                                        (d_0_helpers_).BoostTokenLogits(lm, d_18_selectContext_, _dafny.BigRational('3e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_22_next_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = (lm).ChooseNextToken()
                                d_22_next_ = out14_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_22_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_23_appendedGenerated2_: _dafny.Seq
                                    d_24_appendedInside2_: bool
                                    d_25_appendedCurrent2_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                    d_23_appendedGenerated2_ = out15_
                                    d_24_appendedInside2_ = out16_
                                    d_25_appendedCurrent2_ = out17_
                                    generated = d_23_appendedGenerated2_
                                    insideConstrainedOut = d_24_appendedInside2_
                                    currentConstrainedOut = d_25_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

