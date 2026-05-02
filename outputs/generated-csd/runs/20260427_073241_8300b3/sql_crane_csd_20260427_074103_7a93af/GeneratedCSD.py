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
        d_1_fromContext_: _dafny.Seq
        d_1_fromContext_ = _dafny.SeqWithoutIsStrInference([])
        d_2_selectContext_: _dafny.Seq
        d_2_selectContext_ = _dafny.SeqWithoutIsStrInference([])
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
                            d_1_fromContext_ = _dafny.SeqWithoutIsStrInference([])
                            d_2_selectContext_ = _dafny.SeqWithoutIsStrInference([])
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
                            d_4_steps_ = (d_4_steps_) + (1)
                            d_1_fromContext_ = _dafny.SeqWithoutIsStrInference([])
                            d_2_selectContext_ = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            out7_: _dafny.Seq
                            out7_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")))
                            d_1_fromContext_ = out7_
                            out8_: _dafny.Seq
                            out8_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")))
                            d_2_selectContext_ = out8_
                            d_14_stablePrefix_: _dafny.Seq
                            d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_15_constrainedPrompt_: _dafny.Seq
                            d_15_constrainedPrompt_ = (prompt) + (d_14_stablePrefix_)
                            d_16_validCount_: int
                            out9_: int
                            out9_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_16_validCount_ = out9_
                            if (d_16_validCount_) <= (d_3_narrowThreshold_):
                                d_17_next_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_17_next_ = out10_
                                d_4_steps_ = (d_4_steps_) + (1)
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
                                out14_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, 20, eosToken)
                                d_21_candidates_ = out14_
                                d_22_scoped_: _dafny.Seq
                                out15_: _dafny.Seq
                                out15_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_21_candidates_, d_1_fromContext_)
                                d_22_scoped_ = out15_
                                d_23_focused_: _dafny.Seq
                                out16_: _dafny.Seq
                                out16_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_21_candidates_, d_2_selectContext_)
                                d_23_focused_ = out16_
                                if (len(d_22_scoped_)) > (0):
                                    (d_0_helpers_).BoostTokenLogits(lm, d_22_scoped_, _dafny.BigRational('5e0'))
                                if (len(d_23_focused_)) > (0):
                                    (d_0_helpers_).BoostTokenLogits(lm, d_23_focused_, _dafny.BigRational('3e0'))
                                d_24_symbolBudget_: int
                                d_24_symbolBudget_ = (maxSteps) - (d_4_steps_)
                                d_25_symbolOut_: _dafny.Seq
                                d_26_hitEos_: bool
                                d_27_stepsUsed2_: int
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: int
                                out17_, out18_, out19_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, d_24_symbolBudget_, eosToken)
                                d_25_symbolOut_ = out17_
                                d_26_hitEos_ = out18_
                                d_27_stepsUsed2_ = out19_
                                generated = (d_14_stablePrefix_) + (d_25_symbolOut_)
                                currentConstrainedOut = d_25_symbolOut_
                                d_4_steps_ = (d_4_steps_) + (d_27_stepsUsed2_)
                                if d_26_hitEos_:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_4_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

