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
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        d_3_scopeKeyword_: _dafny.Seq
        d_3_scopeKeyword_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))
        d_4_flatPreferred_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_4_flatPreferred_ = out0_
        d_5_semanticContext_: _dafny.Seq
        d_5_semanticContext_ = _dafny.SeqWithoutIsStrInference([])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_chunkBudget_: int
                        d_6_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_7_chunkedGenerated_: _dafny.Seq
                        d_8_stoppedOpen_: bool
                        d_9_stoppedEos_: bool
                        d_10_stepsUsed_: int
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: bool
                        out4_: int
                        out1_, out2_, out3_, out4_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_7_chunkedGenerated_ = out1_
                        d_8_stoppedOpen_ = out2_
                        d_9_stoppedEos_ = out3_
                        d_10_stepsUsed_ = out4_
                        generated = d_7_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_10_stepsUsed_)
                        if d_9_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_8_stoppedOpen_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_5_semanticContext_ = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_11_isComplete_: bool
                        d_11_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_11_isComplete_:
                            d_12_closedGenerated_: _dafny.Seq
                            d_13_closedInside_: bool
                            d_14_closedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_12_closedGenerated_ = out5_
                            d_13_closedInside_ = out6_
                            d_14_closedCurrent_ = out7_
                            generated = d_12_closedGenerated_
                            insideConstrainedOut = d_13_closedInside_
                            currentConstrainedOut = d_14_closedCurrent_
                            d_5_semanticContext_ = _dafny.SeqWithoutIsStrInference([])
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            out8_: _dafny.Seq
                            out8_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_3_scopeKeyword_)
                            d_5_semanticContext_ = out8_
                            d_15_constrainedPrompt_: _dafny.Seq
                            d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_16_validCount_: int
                            out9_: int
                            out9_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_16_validCount_ = out9_
                            if (d_16_validCount_) <= (d_2_narrowThreshold_):
                                d_17_next_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_17_next_ = out10_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_17_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_18_appendedGenerated_: _dafny.Seq
                                    d_19_appendedInside_: bool
                                    d_20_appendedCurrent_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                    d_18_appendedGenerated_ = out11_
                                    d_19_appendedInside_ = out12_
                                    d_20_appendedCurrent_ = out13_
                                    generated = d_18_appendedGenerated_
                                    insideConstrainedOut = d_19_appendedInside_
                                    currentConstrainedOut = d_20_appendedCurrent_
                            elif True:
                                (lm).GenerateLogits((d_15_constrainedPrompt_) + (currentConstrainedOut))
                                d_21_candidates_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, 30, eosToken)
                                d_21_candidates_ = out14_
                                if (len(d_4_flatPreferred_)) > (0):
                                    d_22_preferred_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out15_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_21_candidates_, d_4_flatPreferred_)
                                    d_22_preferred_ = out15_
                                    if (len(d_22_preferred_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_22_preferred_, _dafny.BigRational('5e0'))
                                if (len(d_5_semanticContext_)) > (0):
                                    d_23_focused_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out16_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_21_candidates_, d_5_semanticContext_)
                                    d_23_focused_ = out16_
                                    if (len(d_23_focused_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_23_focused_, _dafny.BigRational('6e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_24_next_: _dafny.Seq
                                out17_: _dafny.Seq
                                out17_ = (lm).ChooseNextToken()
                                d_24_next_ = out17_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_24_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_25_appendedGenerated_: _dafny.Seq
                                    d_26_appendedInside_: bool
                                    d_27_appendedCurrent_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                                    d_25_appendedGenerated_ = out18_
                                    d_26_appendedInside_ = out19_
                                    d_27_appendedCurrent_ = out20_
                                    generated = d_25_appendedGenerated_
                                    insideConstrainedOut = d_26_appendedInside_
                                    currentConstrainedOut = d_27_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

