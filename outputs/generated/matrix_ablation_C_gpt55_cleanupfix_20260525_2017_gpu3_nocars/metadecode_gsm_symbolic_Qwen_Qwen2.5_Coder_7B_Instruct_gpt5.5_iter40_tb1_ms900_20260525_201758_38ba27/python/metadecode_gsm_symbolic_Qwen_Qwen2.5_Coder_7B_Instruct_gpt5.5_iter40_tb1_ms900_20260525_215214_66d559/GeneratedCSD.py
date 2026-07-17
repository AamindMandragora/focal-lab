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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem carefully in words first, but use exactly one visible symbolic span, only for the final answer: Final answer: <<expression>>. Do not put intermediate calculations inside << >>. Preserve variable names exactly, including underscores such as k_2 and n_1. The final expression should include all required terms from the story. Use int((part)/(whole) * 100) for requested integer percentages and // for whole-number counts, trips, leftovers, or ratios when appropriate.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openedFinal_: bool
        d_2_openedFinal_ = insideConstrained
        d_3_finalArmed_: bool
        d_3_finalArmed_ = insideConstrained
        d_4_forceOpenNext_: bool
        d_4_forceOpenNext_ = False
        d_5_forceFinalAfter_: int
        d_5_forceFinalAfter_ = 180
        d_6_penaltyTokens_: _dafny.Seq
        d_6_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " !")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!!!!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " <<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " >>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_4_forceOpenNext_) or ((not(d_2_openedFinal_)) and ((d_1_steps_) >= (d_5_forceFinalAfter_))):
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
                            d_2_openedFinal_ = True
                            d_3_finalArmed_ = True
                            d_4_forceOpenNext_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_10_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_10_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_10_next_) == (eosToken):
                                if (not(d_2_openedFinal_)) and ((d_1_steps_) < (maxSteps)):
                                    d_3_finalArmed_ = True
                                    d_4_forceOpenNext_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif ((((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " <<"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<<<"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " <<<<")))):
                                if d_3_finalArmed_:
                                    d_4_forceOpenNext_ = True
                            elif (((((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " >>"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>>>"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " >>>>"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " >>>")))):
                                pass
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                                if ((((((((((((((((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Final"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " answer"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Answer"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " therefore"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Thus"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " thus"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "thus"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "So"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " so")))):
                                    d_3_finalArmed_ = True
                                if ((((((((((((((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final:")))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final:"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer:"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " answer:"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final answer:"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final answer:"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final answer:"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final answer:"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer is"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " answer is"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer is"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Answer is"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final answer is"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final answer is")))):
                                    d_3_finalArmed_ = True
                                    d_4_forceOpenNext_ = True
                                if ((d_3_finalArmed_) and (not(d_2_openedFinal_))) and (((((((((((((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " :"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ="))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " is"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " are"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " equals"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " is:"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " answer is"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Answer is"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final answer is"))))) or ((d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final answer is"))))):
                                    d_4_forceOpenNext_ = True
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
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_14_constrainedPrompt_: _dafny.Seq
                        d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_15_nextConstrained_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_6_penaltyTokens_, _dafny.BigRational('5e0'), 24, eosToken)
                        d_15_nextConstrained_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_15_nextConstrained_) == (eosToken):
                            if (d_1_steps_) < (maxSteps):
                                d_16_candidates_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, 6, eosToken)
                                d_16_candidates_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_17_alt_: _dafny.Seq
                                d_17_alt_ = (d_16_candidates_)[0]
                                if ((d_17_alt_) == (eosToken)) and ((len(d_16_candidates_)) > (1)):
                                    d_17_alt_ = (d_16_candidates_)[1]
                                if ((d_17_alt_) == (eosToken)) and ((len(d_16_candidates_)) > (2)):
                                    d_17_alt_ = (d_16_candidates_)[2]
                                if ((d_17_alt_) == (eosToken)) and ((len(d_16_candidates_)) > (3)):
                                    d_17_alt_ = (d_16_candidates_)[3]
                                if ((d_17_alt_) == (eosToken)) and ((len(d_16_candidates_)) > (4)):
                                    d_17_alt_ = (d_16_candidates_)[4]
                                if ((d_17_alt_) == (eosToken)) and ((len(d_16_candidates_)) > (5)):
                                    d_17_alt_ = (d_16_candidates_)[5]
                                if (d_17_alt_) != (eosToken):
                                    d_18_appendedGenerated_: _dafny.Seq
                                    d_19_appendedInside_: bool
                                    d_20_appendedCurrent_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_alt_)
                                    d_18_appendedGenerated_ = out9_
                                    d_19_appendedInside_ = out10_
                                    d_20_appendedCurrent_ = out11_
                                    generated = d_18_appendedGenerated_
                                    insideConstrainedOut = d_19_appendedInside_
                                    currentConstrainedOut = d_20_appendedCurrent_
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_21_appendedGenerated2_: _dafny.Seq
                            d_22_appendedInside2_: bool
                            d_23_appendedCurrent2_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_nextConstrained_)
                            d_21_appendedGenerated2_ = out12_
                            d_22_appendedInside2_ = out13_
                            d_23_appendedCurrent2_ = out14_
                            generated = d_21_appendedGenerated2_
                            insideConstrainedOut = d_22_appendedInside2_
                            currentConstrainedOut = d_23_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

