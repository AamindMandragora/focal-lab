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
        d_1_schemaFocus_: _dafny.Seq
        d_1_schemaFocus_ = _dafny.SeqWithoutIsStrInference([])
        d_2_flatPreferred_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_2_flatPreferred_ = out0_
        d_3_scopeKeyword_: _dafny.Seq
        d_3_scopeKeyword_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))
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
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: bool
                        out4_: int
                        out1_, out2_, out3_, out4_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_6_chunkedGenerated_ = out1_
                        d_7_stoppedOpen_ = out2_
                        d_8_stoppedEos_ = out3_
                        d_9_stepsUsed_ = out4_
                        generated = d_6_chunkedGenerated_
                        d_4_steps_ = (d_4_steps_) + (d_9_stepsUsed_)
                        if d_8_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_7_stoppedOpen_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_1_schemaFocus_ = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_10_complete_: bool
                        d_10_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_10_complete_:
                            d_11_closedGenerated_: _dafny.Seq
                            d_12_closedInside_: bool
                            d_13_closedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_11_closedGenerated_ = out5_
                            d_12_closedInside_ = out6_
                            d_13_closedCurrent_ = out7_
                            generated = d_11_closedGenerated_
                            insideConstrainedOut = d_12_closedInside_
                            currentConstrainedOut = d_13_closedCurrent_
                            d_1_schemaFocus_ = _dafny.SeqWithoutIsStrInference([])
                            d_4_steps_ = (d_4_steps_) + (1)
                        elif True:
                            out8_: _dafny.Seq
                            out8_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_3_scopeKeyword_)
                            d_1_schemaFocus_ = out8_
                            d_14_stablePrefix_: _dafny.Seq
                            d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_15_constrainedPrompt_: _dafny.Seq
                            d_15_constrainedPrompt_ = (prompt) + (d_14_stablePrefix_)
                            (lm).GenerateLogits((d_15_constrainedPrompt_) + (currentConstrainedOut))
                            d_16_candidates_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, 30, eosToken)
                            d_16_candidates_ = out9_
                            if (len(d_1_schemaFocus_)) > (0):
                                d_17_focused_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_16_candidates_, d_1_schemaFocus_)
                                d_17_focused_ = out10_
                                if (len(d_17_focused_)) > (0):
                                    (d_0_helpers_).BoostTokenLogits(lm, d_17_focused_, _dafny.BigRational('8e0'))
                            if (len(d_2_flatPreferred_)) > (0):
                                d_18_preferred_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_16_candidates_, d_2_flatPreferred_)
                                d_18_preferred_ = out11_
                                if (len(d_18_preferred_)) > (0):
                                    (d_0_helpers_).BoostTokenLogits(lm, d_18_preferred_, _dafny.BigRational('4e0'))
                            if ((len(d_1_schemaFocus_)) > (0)) and ((len(d_2_flatPreferred_)) > (0)):
                                d_19_doublyPreferred0_: _dafny.Seq
                                out12_: _dafny.Seq
                                out12_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_16_candidates_, d_1_schemaFocus_)
                                d_19_doublyPreferred0_ = out12_
                                d_20_doublyPreferred_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_19_doublyPreferred0_, d_2_flatPreferred_)
                                d_20_doublyPreferred_ = out13_
                                if (len(d_20_doublyPreferred_)) > (0):
                                    (d_0_helpers_).BoostTokenLogits(lm, d_20_doublyPreferred_, _dafny.BigRational('3e0'))
                            (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                            d_21_next_: _dafny.Seq
                            out14_: _dafny.Seq
                            out14_ = (lm).ChooseNextToken()
                            d_21_next_ = out14_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_4_steps_ = (d_4_steps_) + (1)
                            if (d_21_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_22_appendedGenerated_: _dafny.Seq
                                d_23_appendedInside_: bool
                                d_24_appendedCurrent_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                d_22_appendedGenerated_ = out15_
                                d_23_appendedInside_ = out16_
                                d_24_appendedCurrent_ = out17_
                                generated = d_22_appendedGenerated_
                                insideConstrainedOut = d_23_appendedInside_
                                currentConstrainedOut = d_24_appendedCurrent_
                    pass
            pass
        cost = d_4_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

