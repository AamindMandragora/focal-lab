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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the symbolic math word problem concisely. Do a few plain-text reasoning steps, then end with exactly one final visible span: Final answer: <<expression>>. Do not put prose, units, Markdown, or LaTeX inside << >>. Preserve variable names exactly, including underscores such as n_1 and n_2. Use int(...) for percentage/count answers that require an integer, and use // for whole-number division when appropriate. Do not emit any intermediate << >> spans.")))
        if (maxSteps) == (0):
            cost = 0
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_markerArmed_: bool
        d_2_markerArmed_ = False
        d_3_finalCueSeen_: bool
        d_3_finalCueSeen_ = False
        d_4_haveSpan_: bool
        d_4_haveSpan_ = insideConstrained
        d_5_forceAfter_: int
        d_5_forceAfter_ = 90
        d_6_acceptObservedAfter_: int
        d_6_acceptObservedAfter_ = 30
        d_7_penaltyTokens_: _dafny.Seq
        d_7_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " min")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "min")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " max")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "max")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " minimum")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "maximum")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Maximum")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Minimum"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_4_haveSpan_)) and ((d_2_markerArmed_) or ((d_1_steps_) >= (d_5_forceAfter_))):
                            d_8_openedGenerated_: _dafny.Seq
                            d_9_openedInside_: bool
                            d_10_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_8_openedGenerated_ = out0_
                            d_9_openedInside_ = out1_
                            d_10_openedCurrent_ = out2_
                            generated = d_8_openedGenerated_
                            insideConstrainedOut = d_9_openedInside_
                            currentConstrainedOut = d_10_openedCurrent_
                            d_2_markerArmed_ = False
                            d_3_finalCueSeen_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_11_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_11_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_11_next_) == (eosToken):
                                if (not(d_4_haveSpan_)) and ((d_1_steps_) < (maxSteps)):
                                    d_2_markerArmed_ = True
                                    d_3_finalCueSeen_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif (d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                if (not(d_4_haveSpan_)) and ((d_3_finalCueSeen_) or ((d_1_steps_) >= (d_6_acceptObservedAfter_))):
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_markerArmed_ = False
                                    d_3_finalCueSeen_ = True
                            elif (d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                pass
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                if (not(d_4_haveSpan_)) and (((((((((((((((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " answer"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Answer"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer:"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " answer:"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Final"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " therefore"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Thus"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " thus"))))):
                                    d_2_markerArmed_ = True
                                    d_3_finalCueSeen_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_12_closedGenerated_: _dafny.Seq
                        d_13_closedInside_: bool
                        d_14_closedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_12_closedGenerated_ = out4_
                        d_13_closedInside_ = out5_
                        d_14_closedCurrent_ = out6_
                        generated = d_12_closedGenerated_
                        insideConstrainedOut = d_13_closedInside_
                        currentConstrainedOut = d_14_closedCurrent_
                        d_4_haveSpan_ = True
                        d_2_markerArmed_ = False
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_16_nextConstrained_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_7_penaltyTokens_, _dafny.BigRational('4e0'), 12, eosToken)
                        d_16_nextConstrained_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_16_nextConstrained_) != (eosToken):
                            d_17_valid_: bool
                            out8_: bool
                            out8_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_16_nextConstrained_)
                            d_17_valid_ = out8_
                            if d_17_valid_:
                                d_18_appendedGenerated_: _dafny.Seq
                                d_19_appendedInside_: bool
                                d_20_appendedCurrent_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_nextConstrained_)
                                d_18_appendedGenerated_ = out9_
                                d_19_appendedInside_ = out10_
                                d_20_appendedCurrent_ = out11_
                                generated = d_18_appendedGenerated_
                                insideConstrainedOut = d_19_appendedInside_
                                currentConstrainedOut = d_20_appendedCurrent_
                        elif (d_1_steps_) < (maxSteps):
                            d_21_candidates_: _dafny.Seq
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, 4, eosToken)
                            d_21_candidates_ = out12_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_22_alt_: _dafny.Seq
                            d_22_alt_ = (d_21_candidates_)[0]
                            if ((d_22_alt_) == (eosToken)) and ((len(d_21_candidates_)) > (1)):
                                d_22_alt_ = (d_21_candidates_)[1]
                            if (d_22_alt_) != (eosToken):
                                d_23_appendedGenerated2_: _dafny.Seq
                                d_24_appendedInside2_: bool
                                d_25_appendedCurrent2_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_alt_)
                                d_23_appendedGenerated2_ = out13_
                                d_24_appendedInside2_ = out14_
                                d_25_appendedCurrent2_ = out15_
                                generated = d_23_appendedGenerated2_
                                insideConstrainedOut = d_24_appendedInside2_
                                currentConstrainedOut = d_25_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

