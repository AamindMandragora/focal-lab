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
        d_2_narrowThreshold_ = 8
        d_3_schemaBias_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_3_schemaBias_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_remaining_: int
                        d_4_remaining_ = (maxSteps) - (d_1_steps_)
                        d_5_chunkGenerated_: _dafny.Seq
                        d_6_stoppedOnOpenSpan_: bool
                        d_7_stoppedOnEos_: bool
                        d_8_stepsUsed_: int
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: bool
                        out4_: int
                        out1_, out2_, out3_, out4_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_remaining_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_5_chunkGenerated_ = out1_
                        d_6_stoppedOnOpenSpan_ = out2_
                        d_7_stoppedOnEos_ = out3_
                        d_8_stepsUsed_ = out4_
                        generated = d_5_chunkGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                        if d_7_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif True:
                            if d_6_stoppedOnOpenSpan_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif True:
                                raise _dafny.Break("0")
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_9_closedGenerated_: _dafny.Seq
                            d_10_closedInside_: bool
                            d_11_closedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_9_closedGenerated_ = out5_
                            d_10_closedInside_ = out6_
                            d_11_closedCurrent_ = out7_
                            generated = d_9_closedGenerated_
                            insideConstrainedOut = d_10_closedInside_
                            currentConstrainedOut = d_11_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_12_stablePrefix_: _dafny.Seq
                            d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_13_constrainedPrompt_: _dafny.Seq
                            d_13_constrainedPrompt_ = (prompt) + (d_12_stablePrefix_)
                            d_14_validCount_: int
                            out8_: int
                            out8_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_14_validCount_ = out8_
                            if (d_14_validCount_) <= (d_2_narrowThreshold_):
                                d_15_next_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_15_next_ = out9_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_15_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_16_appendedGenerated_: _dafny.Seq
                                    d_17_appendedInside_: bool
                                    d_18_appendedCurrent_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                    d_16_appendedGenerated_ = out10_
                                    d_17_appendedInside_ = out11_
                                    d_18_appendedCurrent_ = out12_
                                    generated = d_16_appendedGenerated_
                                    insideConstrainedOut = d_17_appendedInside_
                                    currentConstrainedOut = d_18_appendedCurrent_
                            elif True:
                                (lm).GenerateLogits((d_13_constrainedPrompt_) + (currentConstrainedOut))
                                d_19_candidates_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, 64, eosToken)
                                d_19_candidates_ = out13_
                                d_20_schemaPreferred_: _dafny.Seq
                                out14_: _dafny.Seq
                                out14_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_19_candidates_, d_3_schemaBias_)
                                d_20_schemaPreferred_ = out14_
                                if (len(d_20_schemaPreferred_)) > (0):
                                    (d_0_helpers_).BoostTokenLogits(lm, d_20_schemaPreferred_, _dafny.BigRational('8e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_21_next2_: _dafny.Seq
                                out15_: _dafny.Seq
                                out15_ = (lm).ChooseNextToken()
                                d_21_next2_ = out15_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_21_next2_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_22_appendedGenerated2_: _dafny.Seq
                                    d_23_appendedInside2_: bool
                                    d_24_appendedCurrent2_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next2_)
                                    d_22_appendedGenerated2_ = out16_
                                    d_23_appendedInside2_ = out17_
                                    d_24_appendedCurrent2_ = out18_
                                    generated = d_22_appendedGenerated2_
                                    insideConstrainedOut = d_23_appendedInside2_
                                    currentConstrainedOut = d_24_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

