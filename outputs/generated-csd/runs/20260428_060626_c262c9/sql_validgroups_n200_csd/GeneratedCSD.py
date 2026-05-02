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
        d_2_scopeKeyword_: _dafny.Seq
        d_2_scopeKeyword_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_3_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                            if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_4_complete_: bool
                        d_4_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_4_complete_:
                            d_5_closedGenerated_: _dafny.Seq
                            d_6_closedInside_: bool
                            d_7_closedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_5_closedGenerated_ = out1_
                            d_6_closedInside_ = out2_
                            d_7_closedCurrent_ = out3_
                            generated = d_5_closedGenerated_
                            insideConstrainedOut = d_6_closedInside_
                            currentConstrainedOut = d_7_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_stablePrefix_: _dafny.Seq
                            d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_9_constrainedPrompt_: _dafny.Seq
                            d_9_constrainedPrompt_ = (prompt) + (d_8_stablePrefix_)
                            (lm).GenerateLogits((d_9_constrainedPrompt_) + (currentConstrainedOut))
                            d_10_semanticContext_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_2_scopeKeyword_)
                            d_10_semanticContext_ = out4_
                            if (len(d_10_semanticContext_)) > (0):
                                d_11_topCtx_: _dafny.Seq
                                out5_: _dafny.Seq
                                out5_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                d_11_topCtx_ = out5_
                                d_12_focused_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_11_topCtx_, d_10_semanticContext_)
                                d_12_focused_ = out6_
                                if (len(d_12_focused_)) > (0):
                                    (d_0_helpers_).BoostTokenLogits(lm, d_12_focused_, _dafny.BigRational('6e0'))
                            if (len(validTokenGroups)) > (0):
                                d_13_flatPreferred_: _dafny.Seq
                                out7_: _dafny.Seq
                                out7_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                d_13_flatPreferred_ = out7_
                                d_14_topGrouped_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                d_14_topGrouped_ = out8_
                                d_15_groupedFocus_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_14_topGrouped_, d_13_flatPreferred_)
                                d_15_groupedFocus_ = out9_
                                if (len(d_15_groupedFocus_)) > (0):
                                    (d_0_helpers_).BoostTokenLogits(lm, d_15_groupedFocus_, _dafny.BigRational('4e0'))
                            (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                            d_16_chosen_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (lm).ChooseNextToken()
                            d_16_chosen_ = out10_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_16_chosen_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_17_appendedGenerated_: _dafny.Seq
                                d_18_appendedInside_: bool
                                d_19_appendedCurrent_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_chosen_)
                                d_17_appendedGenerated_ = out11_
                                d_18_appendedInside_ = out12_
                                d_19_appendedCurrent_ = out13_
                                generated = d_17_appendedGenerated_
                                insideConstrainedOut = d_18_appendedInside_
                                currentConstrainedOut = d_19_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

