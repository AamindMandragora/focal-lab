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
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_4_chunkedGenerated_ = out0_
                        d_5_stoppedOpen_ = out1_
                        d_6_stoppedEos_ = out2_
                        d_7_stepsUsed_ = out3_
                        generated = d_4_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                        if d_6_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_5_stoppedOpen_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_8_closedGenerated_: _dafny.Seq
                            d_9_closedInside_: bool
                            d_10_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_8_closedGenerated_ = out4_
                            d_9_closedInside_ = out5_
                            d_10_closedCurrent_ = out6_
                            generated = d_8_closedGenerated_
                            insideConstrainedOut = d_9_closedInside_
                            currentConstrainedOut = d_10_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_11_validCount_: int
                            out7_: int
                            out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_11_validCount_ = out7_
                            d_12_stablePrefix_: _dafny.Seq
                            d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_13_constrainedPrompt_: _dafny.Seq
                            d_13_constrainedPrompt_ = (prompt) + (d_12_stablePrefix_)
                            if ((d_11_validCount_) > (d_2_narrowThreshold_)) and ((0) < (stepTokenBudget)):
                                d_14_symbolOut_: _dafny.Seq
                                d_15_hitEos_: bool
                                d_16_stepsUsed_: int
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: int
                                out8_, out9_, out10_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, stepTokenBudget, eosToken)
                                d_14_symbolOut_ = out8_
                                d_15_hitEos_ = out9_
                                d_16_stepsUsed_ = out10_
                                if (d_16_stepsUsed_) <= ((maxSteps) - (d_1_steps_)):
                                    generated = (d_12_stablePrefix_) + (d_14_symbolOut_)
                                    currentConstrainedOut = d_14_symbolOut_
                                    d_1_steps_ = (d_1_steps_) + (d_16_stepsUsed_)
                                    if d_15_hitEos_:
                                        raise _dafny.Break("0")
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                (lm).GenerateLogits((d_13_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(validTokenGroups)) > (0):
                                    d_17_flatPreferred_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out11_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_17_flatPreferred_ = out11_
                                    d_18_candidates_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out12_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                    d_18_candidates_ = out12_
                                    d_19_preferred_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out13_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_18_candidates_, d_17_flatPreferred_)
                                    d_19_preferred_ = out13_
                                    if (len(d_19_preferred_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_19_preferred_, _dafny.BigRational('6e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_20_next_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = (lm).ChooseNextToken()
                                d_20_next_ = out14_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_20_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_21_appendedGenerated_: _dafny.Seq
                                    d_22_appendedInside_: bool
                                    d_23_appendedCurrent_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                                    d_21_appendedGenerated_ = out15_
                                    d_22_appendedInside_ = out16_
                                    d_23_appendedCurrent_ = out17_
                                    generated = d_21_appendedGenerated_
                                    insideConstrainedOut = d_22_appendedInside_
                                    currentConstrainedOut = d_23_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

