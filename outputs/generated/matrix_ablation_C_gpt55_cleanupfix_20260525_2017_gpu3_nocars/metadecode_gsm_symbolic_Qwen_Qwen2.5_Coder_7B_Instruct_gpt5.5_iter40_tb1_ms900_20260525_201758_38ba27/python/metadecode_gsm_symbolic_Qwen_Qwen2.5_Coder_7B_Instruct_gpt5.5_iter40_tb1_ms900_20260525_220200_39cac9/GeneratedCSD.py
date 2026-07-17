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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem concisely, then give one final visible answer span: Final answer: <<expression>>. Do not wrap intermediate calculations. Preserve variable names exactly, especially underscores such as n_1, n_2, k_2, and k_3; never rewrite n_2 as n2. Include every contribution from the story before giving the final expression. Use int((part)/(whole) * 100) only for requested integer percentages. For whole-number counts, trips, rates, leftovers, and ratios use // integer division when appropriate, not int(...). Avoid LaTeX; output a plain symbolic expression inside << >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openedAny_: bool
        d_2_openedAny_ = insideConstrained
        d_3_finalArmed_: bool
        d_3_finalArmed_ = insideConstrained
        d_4_forceOpenNext_: bool
        d_4_forceOpenNext_ = False
        d_5_forceFinalAfter_: int
        d_5_forceFinalAfter_ = 70
        d_6_hardStop_: int
        d_6_hardStop_ = 210
        d_7_penaltyTokens_: _dafny.Seq
        d_7_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " n2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " n1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "k2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " k2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "k3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " k3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\[")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\]")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " <<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " >>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_2_openedAny_)) and ((d_4_forceOpenNext_) or ((d_1_steps_) >= (d_5_forceFinalAfter_))):
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
                            d_2_openedAny_ = True
                            d_3_finalArmed_ = True
                            d_4_forceOpenNext_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_2_openedAny_) and ((d_1_steps_) >= (d_6_hardStop_)):
                            raise _dafny.Break("0")
                        elif True:
                            d_11_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_11_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_11_next_) == (eosToken):
                                if (not(d_2_openedAny_)) and ((d_1_steps_) < (maxSteps)):
                                    d_3_finalArmed_ = True
                                    d_4_forceOpenNext_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif ((((((((((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " <<"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<<<"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " <<<<"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":<<"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " :<<"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final answer:<<"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final answer:<<"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer:<<"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " answer:<<")))):
                                d_3_finalArmed_ = True
                                if not(d_2_openedAny_):
                                    d_4_forceOpenNext_ = True
                            elif (((((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " >>"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>>>"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " >>>>"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " >>>")))):
                                pass
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                if ((((((((((((((((((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Final"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " answer"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Answer"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " therefore"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Thus"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " thus"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "thus"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Hence"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " hence"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "So"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " so")))):
                                    d_3_finalArmed_ = True
                                if ((((((((((((((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final:")))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final:"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer:"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " answer:"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final answer:"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final answer:"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final answer:"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final answer:"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer is"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " answer is"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer is"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Answer is"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final answer is"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final answer is")))):
                                    d_3_finalArmed_ = True
                                    if not(d_2_openedAny_):
                                        d_4_forceOpenNext_ = True
                                if (((d_3_finalArmed_) and (not(d_2_openedAny_))) and ((d_1_steps_) >= (12))) and (((((((((((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " :"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ="))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " is"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " are"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " equals"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " is:"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " answer is"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Answer is"))))):
                                    d_4_forceOpenNext_ = True
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
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_16_nextConstrained_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_7_penaltyTokens_, _dafny.BigRational('2e0'), 32, eosToken)
                        d_16_nextConstrained_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_16_nextConstrained_) == (eosToken):
                            if (d_1_steps_) < (maxSteps):
                                d_17_candidates_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, 6, eosToken)
                                d_17_candidates_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_18_alt_: _dafny.Seq
                                d_18_alt_ = (d_17_candidates_)[0]
                                if ((d_18_alt_) == (eosToken)) and ((len(d_17_candidates_)) > (1)):
                                    d_18_alt_ = (d_17_candidates_)[1]
                                if ((d_18_alt_) == (eosToken)) and ((len(d_17_candidates_)) > (2)):
                                    d_18_alt_ = (d_17_candidates_)[2]
                                if (d_18_alt_) != (eosToken):
                                    d_19_appendedGenerated_: _dafny.Seq
                                    d_20_appendedInside_: bool
                                    d_21_appendedCurrent_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_alt_)
                                    d_19_appendedGenerated_ = out9_
                                    d_20_appendedInside_ = out10_
                                    d_21_appendedCurrent_ = out11_
                                    generated = d_19_appendedGenerated_
                                    insideConstrainedOut = d_20_appendedInside_
                                    currentConstrainedOut = d_21_appendedCurrent_
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_22_appendedGenerated2_: _dafny.Seq
                            d_23_appendedInside2_: bool
                            d_24_appendedCurrent2_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_nextConstrained_)
                            d_22_appendedGenerated2_ = out12_
                            d_23_appendedInside2_ = out13_
                            d_24_appendedCurrent2_ = out14_
                            generated = d_22_appendedGenerated2_
                            insideConstrainedOut = d_23_appendedInside2_
                            currentConstrainedOut = d_24_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

