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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_fromKeyword_: _dafny.Seq
        d_2_fromKeyword_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))
        d_3_dotToken_: _dafny.Seq
        d_3_dotToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "."))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_5_complete_: bool
                        d_5_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_5_complete_:
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
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_stablePrefix_: _dafny.Seq
                            d_9_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_10_constrainedPrompt_: _dafny.Seq
                            d_10_constrainedPrompt_ = (prompt) + (d_9_stablePrefix_)
                            (lm).GenerateLogits((d_10_constrainedPrompt_) + (currentConstrainedOut))
                            if (len(validTokenGroups)) > (0):
                                (d_0_helpers_).BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'))
                            d_11_fromContext_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_2_fromKeyword_)
                            d_11_fromContext_ = out4_
                            if (len(d_11_fromContext_)) > (0):
                                d_12_anyScopedValid_: bool
                                out5_: bool
                                out5_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_11_fromContext_)
                                d_12_anyScopedValid_ = out5_
                                if d_12_anyScopedValid_:
                                    d_13_topValid_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out6_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                    d_13_topValid_ = out6_
                                    d_14_scopedPreferred_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out7_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_13_topValid_, d_11_fromContext_)
                                    d_14_scopedPreferred_ = out7_
                                    if (len(d_14_scopedPreferred_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_14_scopedPreferred_, _dafny.BigRational('6e0'))
                            d_15_prevBeforeDot_: _dafny.Seq
                            d_16_foundDot_: bool
                            out8_: _dafny.Seq
                            out9_: bool
                            out8_, out9_ = (d_0_helpers_).LastTokenBefore(currentConstrainedOut, d_3_dotToken_)
                            d_15_prevBeforeDot_ = out8_
                            d_16_foundDot_ = out9_
                            if d_16_foundDot_:
                                if (len(d_11_fromContext_)) > (0):
                                    d_17_topValid2_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                    d_17_topValid2_ = out10_
                                    d_18_tableLike_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out11_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_17_topValid2_, d_11_fromContext_)
                                    d_18_tableLike_ = out11_
                                    if (len(d_18_tableLike_)) > (0):
                                        (d_0_helpers_).PenalizeTokenLogits(lm, d_18_tableLike_, _dafny.BigRational('2e0'))
                            (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                            d_19_sampled_: _dafny.Seq
                            out12_: _dafny.Seq
                            out12_ = (lm).ChooseNextToken()
                            d_19_sampled_ = out12_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_19_sampled_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_20_appendedGenerated_: _dafny.Seq
                                d_21_appendedInside_: bool
                                d_22_appendedCurrent_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_sampled_)
                                d_20_appendedGenerated_ = out13_
                                d_21_appendedInside_ = out14_
                                d_22_appendedCurrent_ = out15_
                                generated = d_20_appendedGenerated_
                                insideConstrainedOut = d_21_appendedInside_
                                currentConstrainedOut = d_22_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

