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
        d_3_fromKeyword_: _dafny.Seq
        d_3_fromKeyword_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))
        d_4_selectKeyword_: _dafny.Seq
        d_4_selectKeyword_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT"))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_chunkBudget_: int
                        d_5_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_6_chunkedGenerated_: _dafny.Seq
                        d_7_stoppedOnOpenSpan_: bool
                        d_8_stoppedOnEos_: bool
                        d_9_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_6_chunkedGenerated_ = out0_
                        d_7_stoppedOnOpenSpan_ = out1_
                        d_8_stoppedOnEos_ = out2_
                        d_9_stepsUsed_ = out3_
                        generated = d_6_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                        if d_8_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_7_stoppedOnOpenSpan_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_10_isComplete_: bool
                        d_10_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_10_isComplete_:
                            d_11_closedGenerated_: _dafny.Seq
                            d_12_closedInside_: bool
                            d_13_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_11_closedGenerated_ = out4_
                            d_12_closedInside_ = out5_
                            d_13_closedCurrent_ = out6_
                            generated = d_11_closedGenerated_
                            insideConstrainedOut = d_12_closedInside_
                            currentConstrainedOut = d_13_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_14_constrainedPrompt_: _dafny.Seq
                            d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_15_validCount_: int
                            out7_: int
                            out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_15_validCount_ = out7_
                            if (d_15_validCount_) <= (d_2_narrowThreshold_):
                                d_16_next_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_16_next_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_16_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_17_appendedGenerated_: _dafny.Seq
                                    d_18_appendedInside_: bool
                                    d_19_appendedCurrent_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                                    d_17_appendedGenerated_ = out9_
                                    d_18_appendedInside_ = out10_
                                    d_19_appendedCurrent_ = out11_
                                    generated = d_17_appendedGenerated_
                                    insideConstrainedOut = d_18_appendedInside_
                                    currentConstrainedOut = d_19_appendedCurrent_
                            elif True:
                                d_20_fromContext_: _dafny.Seq
                                out12_: _dafny.Seq
                                out12_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_3_fromKeyword_)
                                d_20_fromContext_ = out12_
                                d_21_selectContext_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_4_selectKeyword_)
                                d_21_selectContext_ = out13_
                                (lm).GenerateLogits((d_14_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(validTokenGroups)) > (0):
                                    (d_0_helpers_).BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'))
                                if (len(d_20_fromContext_)) > (0):
                                    d_22_topFrom_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out14_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                    d_22_topFrom_ = out14_
                                    d_23_focusedFrom_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out15_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_22_topFrom_, d_20_fromContext_)
                                    d_23_focusedFrom_ = out15_
                                    if (len(d_23_focusedFrom_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_23_focusedFrom_, _dafny.BigRational('8e0'))
                                if (len(d_21_selectContext_)) > (0):
                                    d_24_topSelect_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out16_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                    d_24_topSelect_ = out16_
                                    d_25_focusedSelect_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out17_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_24_topSelect_, d_21_selectContext_)
                                    d_25_focusedSelect_ = out17_
                                    if (len(d_25_focusedSelect_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_25_focusedSelect_, _dafny.BigRational('5e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_26_nextWide_: _dafny.Seq
                                out18_: _dafny.Seq
                                out18_ = (lm).ChooseNextToken()
                                d_26_nextWide_ = out18_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_26_nextWide_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_27_appendedGenerated2_: _dafny.Seq
                                    d_28_appendedInside2_: bool
                                    d_29_appendedCurrent2_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out20_: bool
                                    out21_: _dafny.Seq
                                    out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_nextWide_)
                                    d_27_appendedGenerated2_ = out19_
                                    d_28_appendedInside2_ = out20_
                                    d_29_appendedCurrent2_ = out21_
                                    generated = d_27_appendedGenerated2_
                                    insideConstrainedOut = d_28_appendedInside2_
                                    currentConstrainedOut = d_29_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

