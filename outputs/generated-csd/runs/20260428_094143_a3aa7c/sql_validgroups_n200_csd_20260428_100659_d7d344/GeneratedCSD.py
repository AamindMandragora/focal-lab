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
        d_2_flatPreferred_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_2_flatPreferred_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_chunkBudget_: int
                        d_3_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_4_chunkedGenerated_: _dafny.Seq
                        d_5_stoppedOpen_: bool
                        d_6_stoppedEos_: bool
                        d_7_stepsUsed_: int
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: bool
                        out4_: int
                        out1_, out2_, out3_, out4_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_4_chunkedGenerated_ = out1_
                        d_5_stoppedOpen_ = out2_
                        d_6_stoppedEos_ = out3_
                        d_7_stepsUsed_ = out4_
                        generated = d_4_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                        if d_6_stoppedEos_:
                            raise _dafny.Break("0")
                        elif True:
                            if d_5_stoppedOpen_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_8_constrainedPrompt_: _dafny.Seq
                        d_8_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_9_isComplete_: bool
                        d_9_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_9_isComplete_:
                            (lm).GenerateLogits((d_8_constrainedPrompt_) + (currentConstrainedOut))
                            (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                            d_10_prevTok_: _dafny.Seq
                            d_11_foundPrev_: bool
                            out5_: _dafny.Seq
                            out6_: bool
                            out5_, out6_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                            d_10_prevTok_ = out5_
                            d_11_foundPrev_ = out6_
                            if d_11_foundPrev_:
                                if (d_10_prevTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER"))):
                                    d_12_byValid_: bool
                                    out7_: bool
                                    out7_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BY")))
                                    d_12_byValid_ = out7_
                                    if d_12_byValid_:
                                        if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BY"))) in ((lm).Tokens):
                                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BY"))]), _dafny.BigRational('1e1'))
                            d_13_next_: _dafny.Seq
                            out8_: _dafny.Seq
                            out8_ = (lm).ChooseNextToken()
                            d_13_next_ = out8_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_13_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_14_closedGenerated2_: _dafny.Seq
                                d_15_closedInside2_: bool
                                d_16_closedCurrent2_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_14_closedGenerated2_ = out9_
                                d_15_closedInside2_ = out10_
                                d_16_closedCurrent2_ = out11_
                                generated = d_14_closedGenerated2_
                                insideConstrainedOut = d_15_closedInside2_
                                currentConstrainedOut = d_16_closedCurrent2_
                        elif True:
                            (lm).GenerateLogits((d_8_constrainedPrompt_) + (currentConstrainedOut))
                            d_17_candidates_: _dafny.Seq
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, 40, eosToken)
                            d_17_candidates_ = out12_
                            if (len(d_2_flatPreferred_)) > (0):
                                d_18_preferred_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_17_candidates_, d_2_flatPreferred_)
                                d_18_preferred_ = out13_
                                if (len(d_18_preferred_)) > (0):
                                    (d_0_helpers_).BoostTokenLogits(lm, d_18_preferred_, _dafny.BigRational('2e0'))
                            d_19_prevTok2_: _dafny.Seq
                            d_20_foundPrev2_: bool
                            out14_: _dafny.Seq
                            out15_: bool
                            out14_, out15_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                            d_19_prevTok2_ = out14_
                            d_20_foundPrev2_ = out15_
                            if d_20_foundPrev2_:
                                if (d_19_prevTok2_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER"))):
                                    d_21_byValid2_: bool
                                    out16_: bool
                                    out16_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BY")))
                                    d_21_byValid2_ = out16_
                                    if d_21_byValid2_:
                                        if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BY"))) in ((lm).Tokens):
                                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BY"))]), _dafny.BigRational('6e0'))
                            (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                            d_22_next2_: _dafny.Seq
                            out17_: _dafny.Seq
                            out17_ = (lm).ChooseNextToken()
                            d_22_next2_ = out17_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_22_next2_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_23_appendedGenerated2_: _dafny.Seq
                                d_24_appendedInside2_: bool
                                d_25_appendedCurrent2_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next2_)
                                d_23_appendedGenerated2_ = out18_
                                d_24_appendedInside2_ = out19_
                                d_25_appendedCurrent2_ = out20_
                                generated = d_23_appendedGenerated2_
                                insideConstrainedOut = d_24_appendedInside2_
                                currentConstrainedOut = d_25_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

