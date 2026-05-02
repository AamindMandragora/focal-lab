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
        d_2_openSpanToken_: _dafny.Seq
        d_2_openSpanToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))
        d_3_closeSpanToken_: _dafny.Seq
        d_3_closeSpanToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))
        d_4_flatPreferred_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_4_flatPreferred_ = out0_
        d_5_sqlBias_: _dafny.Seq
        d_5_sqlBias_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BY")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_openSpanToken_]))
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_1_steps_ = (d_1_steps_) + (1)
                        cost = d_1_steps_
                    elif True:
                        d_6_completeNow_: bool
                        d_6_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_6_completeNow_:
                            if (d_1_steps_) < (maxSteps):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_closeSpanToken_]))
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_1_steps_ = (d_1_steps_) + (1)
                                cost = d_1_steps_
                            raise _dafny.Break("0")
                        elif True:
                            d_7_stablePrefix_: _dafny.Seq
                            d_7_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_8_constrainedPrompt_: _dafny.Seq
                            d_8_constrainedPrompt_ = (prompt) + (d_7_stablePrefix_)
                            d_9_validCount_: int
                            out1_: int
                            out1_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_9_validCount_ = out1_
                            if (((stepTokenBudget) == (0)) or (((d_1_steps_) + (stepTokenBudget)) > (maxSteps))) or ((d_9_validCount_) <= (8)):
                                (lm).GenerateLogits((d_8_constrainedPrompt_) + (currentConstrainedOut))
                                d_10_candidates_: _dafny.Seq
                                out2_: _dafny.Seq
                                out2_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, 32, eosToken)
                                d_10_candidates_ = out2_
                                d_11_structural_: _dafny.Seq
                                out3_: _dafny.Seq
                                out3_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_10_candidates_, d_5_sqlBias_)
                                d_11_structural_ = out3_
                                if (len(d_11_structural_)) > (0):
                                    (d_0_helpers_).BoostTokenLogits(lm, d_11_structural_, _dafny.BigRational('6e0'))
                                if (len(d_4_flatPreferred_)) > (0):
                                    d_12_preferred_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out4_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_10_candidates_, d_4_flatPreferred_)
                                    d_12_preferred_ = out4_
                                    if (len(d_12_preferred_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_12_preferred_, _dafny.BigRational('3e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_13_next_: _dafny.Seq
                                out5_: _dafny.Seq
                                out5_ = (lm).ChooseNextToken()
                                d_13_next_ = out5_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                cost = d_1_steps_
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
                                d_17_symbolOut_: _dafny.Seq
                                d_18_hitEos_: bool
                                d_19_symbolStepsUsed_: int
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: int
                                out9_, out10_, out11_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, stepTokenBudget, eosToken)
                                d_17_symbolOut_ = out9_
                                d_18_hitEos_ = out10_
                                d_19_symbolStepsUsed_ = out11_
                                generated = (d_7_stablePrefix_) + (d_17_symbolOut_)
                                currentConstrainedOut = d_17_symbolOut_
                                d_1_steps_ = (d_1_steps_) + (d_19_symbolStepsUsed_)
                                cost = d_1_steps_
                                if d_18_hitEos_:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

