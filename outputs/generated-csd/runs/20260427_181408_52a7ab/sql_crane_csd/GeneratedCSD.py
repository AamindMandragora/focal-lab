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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, eosToken):
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
        d_1_semanticContext_: _dafny.Seq
        d_1_semanticContext_ = _dafny.SeqWithoutIsStrInference([])
        d_2_scopeKeyword_: _dafny.Seq
        d_2_scopeKeyword_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_steps_: int
        d_4_steps_ = 0
        with _dafny.label("0"):
            while (d_4_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_chunkBudget_: int
                        d_5_chunkBudget_ = (maxSteps) - (d_4_steps_)
                        d_6_chunkedGenerated_: _dafny.Seq
                        d_7_stoppedOpen_: bool
                        d_8_stoppedEos_: bool
                        d_9_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_6_chunkedGenerated_ = out0_
                        d_7_stoppedOpen_ = out1_
                        d_8_stoppedEos_ = out2_
                        d_9_stepsUsed_ = out3_
                        generated = d_6_chunkedGenerated_
                        d_4_steps_ = (d_4_steps_) + (d_9_stepsUsed_)
                        if d_8_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_7_stoppedOpen_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_1_semanticContext_ = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_10_complete_: bool
                        d_10_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_10_complete_:
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
                            d_1_semanticContext_ = _dafny.SeqWithoutIsStrInference([])
                            d_4_steps_ = (d_4_steps_) + (1)
                        elif True:
                            out7_: _dafny.Seq
                            out7_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_2_scopeKeyword_)
                            d_1_semanticContext_ = out7_
                            d_14_validCount_: int
                            out8_: int
                            out8_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_14_validCount_ = out8_
                            d_15_stablePrefix_: _dafny.Seq
                            d_15_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_16_constrainedPrompt_: _dafny.Seq
                            d_16_constrainedPrompt_ = (prompt) + (d_15_stablePrefix_)
                            if (d_14_validCount_) <= (d_3_narrowThreshold_):
                                (lm).GenerateLogits((d_16_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(d_1_semanticContext_)) > (0):
                                    d_17_candidates_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out9_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, 20, eosToken)
                                    d_17_candidates_ = out9_
                                    d_18_focused_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_17_candidates_, d_1_semanticContext_)
                                    d_18_focused_ = out10_
                                    if (len(d_18_focused_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_18_focused_, _dafny.BigRational('6e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_19_next_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = (lm).ChooseNextToken()
                                d_19_next_ = out11_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_4_steps_ = (d_4_steps_) + (1)
                                if (d_19_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_20_appendedGenerated_: _dafny.Seq
                                    d_21_appendedInside_: bool
                                    d_22_appendedCurrent_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                    d_20_appendedGenerated_ = out12_
                                    d_21_appendedInside_ = out13_
                                    d_22_appendedCurrent_ = out14_
                                    generated = d_20_appendedGenerated_
                                    insideConstrainedOut = d_21_appendedInside_
                                    currentConstrainedOut = d_22_appendedCurrent_
                            elif True:
                                d_23_symbolBudget_: int
                                d_23_symbolBudget_ = (maxSteps) - (d_4_steps_)
                                d_24_symbolOut_: _dafny.Seq
                                d_25_hitEos_: bool
                                d_26_stepsUsed2_: int
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: int
                                out15_, out16_, out17_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, d_23_symbolBudget_, eosToken)
                                d_24_symbolOut_ = out15_
                                d_25_hitEos_ = out16_
                                d_26_stepsUsed2_ = out17_
                                generated = (d_15_stablePrefix_) + (d_24_symbolOut_)
                                insideConstrainedOut = True
                                currentConstrainedOut = d_24_symbolOut_
                                d_4_steps_ = (d_4_steps_) + (d_26_stepsUsed2_)
                                if d_25_hitEos_:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_4_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

