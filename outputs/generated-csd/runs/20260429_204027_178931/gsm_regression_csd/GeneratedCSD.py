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
        d_2_preferredFlat_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_2_preferredFlat_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_chunkBudget_: int
                        d_3_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_4_chunkedGenerated_: _dafny.Seq
                        d_5_stoppedOnOpenSpan_: bool
                        d_6_stoppedOnEos_: bool
                        d_7_stepsUsed_: int
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: bool
                        out4_: int
                        out1_, out2_, out3_, out4_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_4_chunkedGenerated_ = out1_
                        d_5_stoppedOnOpenSpan_ = out2_
                        d_6_stoppedOnEos_ = out3_
                        d_7_stepsUsed_ = out4_
                        generated = d_4_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                        if d_6_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_5_stoppedOnOpenSpan_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_8_closedGenerated_: _dafny.Seq
                            d_9_closedInside_: bool
                            d_10_closedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_8_closedGenerated_ = out5_
                            d_9_closedInside_ = out6_
                            d_10_closedCurrent_ = out7_
                            generated = d_8_closedGenerated_
                            insideConstrainedOut = d_9_closedInside_
                            currentConstrainedOut = d_10_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_11_stablePrefix_: _dafny.Seq
                            d_11_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_12_constrainedPrompt_: _dafny.Seq
                            d_12_constrainedPrompt_ = (prompt) + (d_11_stablePrefix_)
                            d_13_sawPlus_: bool
                            d_13_sawPlus_ = (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+"))) in (currentConstrainedOut)
                            d_14_sawMinus_: bool
                            d_14_sawMinus_ = (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-"))) in (currentConstrainedOut)
                            d_15_sawTimes_: bool
                            d_15_sawTimes_ = (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*"))) in (currentConstrainedOut)
                            d_16_sawDivide_: bool
                            d_16_sawDivide_ = (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/"))) in (currentConstrainedOut)
                            d_17_sawEquals_: bool
                            d_17_sawEquals_ = (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))) in (currentConstrainedOut)
                            d_18_arithmeticLike_: bool
                            d_18_arithmeticLike_ = ((((d_13_sawPlus_) or (d_14_sawMinus_)) or (d_15_sawTimes_)) or (d_16_sawDivide_)) or (d_17_sawEquals_)
                            (lm).GenerateLogits((d_12_constrainedPrompt_) + (currentConstrainedOut))
                            if (d_18_arithmeticLike_) and ((len(d_2_preferredFlat_)) > (0)):
                                d_19_anyPreferredValid_: bool
                                out8_: bool
                                out8_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_2_preferredFlat_)
                                d_19_anyPreferredValid_ = out8_
                                if d_19_anyPreferredValid_:
                                    d_20_candidates_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out9_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                    d_20_candidates_ = out9_
                                    d_21_preferredCandidates_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_20_candidates_, d_2_preferredFlat_)
                                    d_21_preferredCandidates_ = out10_
                                    if (len(d_21_preferredCandidates_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_21_preferredCandidates_, _dafny.BigRational('6e0'))
                            (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                            d_22_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (lm).ChooseNextToken()
                            d_22_next_ = out11_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_22_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_23_appendedGenerated_: _dafny.Seq
                                d_24_appendedInside_: bool
                                d_25_appendedCurrent_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                d_23_appendedGenerated_ = out12_
                                d_24_appendedInside_ = out13_
                                d_25_appendedCurrent_ = out14_
                                generated = d_23_appendedGenerated_
                                insideConstrainedOut = d_24_appendedInside_
                                currentConstrainedOut = d_25_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

