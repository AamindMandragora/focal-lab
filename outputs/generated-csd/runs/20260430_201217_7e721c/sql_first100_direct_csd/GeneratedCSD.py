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
        d_2_scopeKeyword_: _dafny.Seq
        d_2_scopeKeyword_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))
        d_3_semanticScope_: _dafny.Seq
        d_3_semanticScope_ = _dafny.SeqWithoutIsStrInference([])
        d_4_narrowThreshold_: int
        d_4_narrowThreshold_ = 12
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
                                d_3_semanticScope_ = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_6_closedGenerated_: _dafny.Seq
                            d_7_closedInside_: bool
                            d_8_closedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_6_closedGenerated_ = out1_
                            d_7_closedInside_ = out2_
                            d_8_closedCurrent_ = out3_
                            generated = d_6_closedGenerated_
                            insideConstrainedOut = d_7_closedInside_
                            currentConstrainedOut = d_8_closedCurrent_
                            d_3_semanticScope_ = _dafny.SeqWithoutIsStrInference([])
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            out4_: _dafny.Seq
                            out4_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_2_scopeKeyword_)
                            d_3_semanticScope_ = out4_
                            d_9_stablePrefix_: _dafny.Seq
                            d_9_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_10_constrainedPrompt_: _dafny.Seq
                            d_10_constrainedPrompt_ = (prompt) + (d_9_stablePrefix_)
                            d_11_validCount_: int
                            out5_: int
                            out5_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_11_validCount_ = out5_
                            if (d_11_validCount_) > (d_4_narrowThreshold_):
                                d_12_next_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_4_narrowThreshold_, eosToken)
                                d_12_next_ = out6_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_12_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_13_appendedGenerated_: _dafny.Seq
                                    d_14_appendedInside_: bool
                                    d_15_appendedCurrent_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                                    d_13_appendedGenerated_ = out7_
                                    d_14_appendedInside_ = out8_
                                    d_15_appendedCurrent_ = out9_
                                    generated = d_13_appendedGenerated_
                                    insideConstrainedOut = d_14_appendedInside_
                                    currentConstrainedOut = d_15_appendedCurrent_
                            elif True:
                                (lm).GenerateLogits((d_10_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(validTokenGroups)) > (0):
                                    (d_0_helpers_).BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'))
                                if (len(d_3_semanticScope_)) > (0):
                                    d_16_flatPreferred_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_16_flatPreferred_ = out10_
                                    d_17_focused_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out11_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_16_flatPreferred_, d_3_semanticScope_)
                                    d_17_focused_ = out11_
                                    if (len(d_17_focused_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_17_focused_, _dafny.BigRational('8e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_18_next_: _dafny.Seq
                                out12_: _dafny.Seq
                                out12_ = (lm).ChooseNextToken()
                                d_18_next_ = out12_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_18_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_19_appendedGenerated_: _dafny.Seq
                                    d_20_appendedInside_: bool
                                    d_21_appendedCurrent_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                                    d_19_appendedGenerated_ = out13_
                                    d_20_appendedInside_ = out14_
                                    d_21_appendedCurrent_ = out15_
                                    generated = d_19_appendedGenerated_
                                    insideConstrainedOut = d_20_appendedInside_
                                    currentConstrainedOut = d_21_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

