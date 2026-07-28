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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the symbolic math word problem step by step in plain text, but use exactly one visible constrained span at the end. Do not wrap intermediate calculations. End with: Final answer: <<expression>>. The expression must contain only the final formula or integer expression, not prose. Preserve all variable names exactly as written, including underscores. Treat variables in braces as symbolic quantities. For 't minutes per d miles over y miles', use y//d*t. For total weights, sums, counts, and trips, include all relevant terms and use // for whole-number division when appropriate.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_markerArmed_: bool
        d_2_markerArmed_ = False
        d_3_finalCueSeen_: bool
        d_3_finalCueSeen_ = False
        d_4_haveSpan_: bool
        d_4_haveSpan_ = insideConstrained
        d_5_forceAfter_: int
        d_5_forceAfter_ = 260
        d_6_penaltyTokens_: _dafny.Seq
        d_6_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "min")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " min")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "max")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " max")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "minimum")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " maximum")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Minimum")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Maximum"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_2_markerArmed_) or ((not(d_4_haveSpan_)) and ((d_1_steps_) >= (d_5_forceAfter_))):
                            d_7_openedGenerated_: _dafny.Seq
                            d_8_openedInside_: bool
                            d_9_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_7_openedGenerated_ = out0_
                            d_8_openedInside_ = out1_
                            d_9_openedCurrent_ = out2_
                            generated = d_7_openedGenerated_
                            insideConstrainedOut = d_8_openedInside_
                            currentConstrainedOut = d_9_openedCurrent_
                            d_2_markerArmed_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_10_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_10_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_10_next_) == (eosToken):
                                if (not(d_4_haveSpan_)) and ((d_1_steps_) < (maxSteps)):
                                    d_2_markerArmed_ = True
                                    d_3_finalCueSeen_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif (d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                if d_3_finalCueSeen_:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_markerArmed_ = False
                            elif (d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                pass
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                                if (not(d_4_haveSpan_)) and (((((((((((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " answer"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Answer"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer:"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " answer:"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " therefore"))))):
                                    d_2_markerArmed_ = True
                                    d_3_finalCueSeen_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
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
                        d_4_haveSpan_ = True
                        d_2_markerArmed_ = False
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_14_constrainedPrompt_: _dafny.Seq
                        d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_15_nextConstrained_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_6_penaltyTokens_, _dafny.BigRational('4e0'), 12, eosToken)
                        d_15_nextConstrained_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_15_nextConstrained_) != (eosToken):
                            d_16_valid_: bool
                            out8_: bool
                            out8_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_15_nextConstrained_)
                            d_16_valid_ = out8_
                            if d_16_valid_:
                                d_17_appendedGenerated_: _dafny.Seq
                                d_18_appendedInside_: bool
                                d_19_appendedCurrent_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_nextConstrained_)
                                d_17_appendedGenerated_ = out9_
                                d_18_appendedInside_ = out10_
                                d_19_appendedCurrent_ = out11_
                                generated = d_17_appendedGenerated_
                                insideConstrainedOut = d_18_appendedInside_
                                currentConstrainedOut = d_19_appendedCurrent_
                        elif (d_1_steps_) < (maxSteps):
                            d_20_candidates_: _dafny.Seq
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, 4, eosToken)
                            d_20_candidates_ = out12_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_21_alt_: _dafny.Seq
                            d_21_alt_ = (d_20_candidates_)[0]
                            if ((d_21_alt_) == (eosToken)) and ((len(d_20_candidates_)) > (1)):
                                d_21_alt_ = (d_20_candidates_)[1]
                            if (d_21_alt_) != (eosToken):
                                d_22_appendedGenerated2_: _dafny.Seq
                                d_23_appendedInside2_: bool
                                d_24_appendedCurrent2_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_alt_)
                                d_22_appendedGenerated2_ = out13_
                                d_23_appendedInside2_ = out14_
                                d_24_appendedCurrent2_ = out15_
                                generated = d_22_appendedGenerated2_
                                insideConstrainedOut = d_23_appendedInside2_
                                currentConstrainedOut = d_24_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

