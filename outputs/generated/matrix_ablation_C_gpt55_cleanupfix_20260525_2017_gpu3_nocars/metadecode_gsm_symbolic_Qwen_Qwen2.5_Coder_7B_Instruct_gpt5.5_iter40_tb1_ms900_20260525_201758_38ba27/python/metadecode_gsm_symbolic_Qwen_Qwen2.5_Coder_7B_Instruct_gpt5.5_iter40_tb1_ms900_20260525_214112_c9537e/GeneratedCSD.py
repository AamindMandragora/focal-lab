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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step, but keep the output concise. Wrap each complete symbolic calculation and the final answer in visible << and >> delimiters. Never type << in the middle of an expression; put the whole expression after the delimiter. Preserve variable names exactly, including underscores such as n_1 and k_2. For percentage answers use int((part)/(whole) * 100). For whole-number counts, trips, ratios, leftovers, or segment counts use // integer division when appropriate. The final line should be Final answer: <<complete expression>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openedAny_: bool
        d_2_openedAny_ = insideConstrained
        d_3_finalArmed_: bool
        d_3_finalArmed_ = insideConstrained
        d_4_forceOpenNext_: bool
        d_4_forceOpenNext_ = False
        d_5_stopAfterClose_: bool
        d_5_stopAfterClose_ = insideConstrained
        d_6_forceFirstAfter_: int
        d_6_forceFirstAfter_ = 90
        d_7_forceFinalAfterAny_: int
        d_7_forceFinalAfterAny_ = 220
        d_8_outsideHardStop_: int
        d_8_outsideHardStop_ = 320
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_4_forceOpenNext_) or ((not(d_2_openedAny_)) and ((d_1_steps_) >= (d_6_forceFirstAfter_)))) or ((d_2_openedAny_) and ((d_1_steps_) >= (d_7_forceFinalAfterAny_))):
                            d_9_openedGenerated_: _dafny.Seq
                            d_10_openedInside_: bool
                            d_11_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_9_openedGenerated_ = out0_
                            d_10_openedInside_ = out1_
                            d_11_openedCurrent_ = out2_
                            generated = d_9_openedGenerated_
                            insideConstrainedOut = d_10_openedInside_
                            currentConstrainedOut = d_11_openedCurrent_
                            d_2_openedAny_ = True
                            d_3_finalArmed_ = True
                            d_4_forceOpenNext_ = False
                            d_5_stopAfterClose_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_2_openedAny_) and ((d_1_steps_) >= (d_8_outsideHardStop_)):
                            d_4_forceOpenNext_ = True
                        elif True:
                            d_12_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_12_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_12_next_) == (eosToken):
                                if (d_1_steps_) < (maxSteps):
                                    d_4_forceOpenNext_ = True
                                    d_3_finalArmed_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif (((((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " >>"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>>>"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " >>>>"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " >>>")))):
                                pass
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                                if ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " <<")))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_openedAny_ = True
                                    d_4_forceOpenNext_ = False
                                    if d_3_finalArmed_:
                                        d_5_stopAfterClose_ = True
                                    elif True:
                                        d_5_stopAfterClose_ = False
                                elif True:
                                    if ((((((((((((((((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Final"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " answer"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Answer"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " therefore"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Thus"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " thus"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "thus"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "So"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " so")))):
                                        d_3_finalArmed_ = True
                                    if ((((((((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final:")))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final:"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer:"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " answer:"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final answer:"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final answer:"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final answer:"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final answer:")))):
                                        d_3_finalArmed_ = True
                                        d_4_forceOpenNext_ = True
                                    if ((d_3_finalArmed_) and ((d_1_steps_) >= (12))) and (((((((((((((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " :"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ="))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " is"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " are"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " equals"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " is:"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " answer is"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Answer is"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final answer is"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final answer is"))))):
                                        d_4_forceOpenNext_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_13_closedGenerated_: _dafny.Seq
                        d_14_closedInside_: bool
                        d_15_closedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_13_closedGenerated_ = out4_
                        d_14_closedInside_ = out5_
                        d_15_closedCurrent_ = out6_
                        generated = d_13_closedGenerated_
                        insideConstrainedOut = d_14_closedInside_
                        currentConstrainedOut = d_15_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_5_stopAfterClose_:
                            raise _dafny.Break("0")
                    elif True:
                        d_16_constrainedPrompt_: _dafny.Seq
                        d_16_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_17_nextConstrained_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 32, eosToken)
                        d_17_nextConstrained_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_17_nextConstrained_) == (eosToken):
                            if (d_1_steps_) < (maxSteps):
                                d_18_candidates_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, 8, eosToken)
                                d_18_candidates_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_19_alt_: _dafny.Seq
                                d_19_alt_ = (d_18_candidates_)[0]
                                if ((d_19_alt_) == (eosToken)) and ((len(d_18_candidates_)) > (1)):
                                    d_19_alt_ = (d_18_candidates_)[1]
                                if ((d_19_alt_) == (eosToken)) and ((len(d_18_candidates_)) > (2)):
                                    d_19_alt_ = (d_18_candidates_)[2]
                                if ((d_19_alt_) == (eosToken)) and ((len(d_18_candidates_)) > (3)):
                                    d_19_alt_ = (d_18_candidates_)[3]
                                if ((d_19_alt_) == (eosToken)) and ((len(d_18_candidates_)) > (4)):
                                    d_19_alt_ = (d_18_candidates_)[4]
                                if (d_19_alt_) != (eosToken):
                                    d_20_appendedGenerated_: _dafny.Seq
                                    d_21_appendedInside_: bool
                                    d_22_appendedCurrent_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_alt_)
                                    d_20_appendedGenerated_ = out9_
                                    d_21_appendedInside_ = out10_
                                    d_22_appendedCurrent_ = out11_
                                    generated = d_20_appendedGenerated_
                                    insideConstrainedOut = d_21_appendedInside_
                                    currentConstrainedOut = d_22_appendedCurrent_
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_23_appendedGenerated2_: _dafny.Seq
                            d_24_appendedInside2_: bool
                            d_25_appendedCurrent2_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_nextConstrained_)
                            d_23_appendedGenerated2_ = out12_
                            d_24_appendedInside2_ = out13_
                            d_25_appendedCurrent2_ = out14_
                            generated = d_23_appendedGenerated2_
                            insideConstrainedOut = d_24_appendedInside2_
                            currentConstrainedOut = d_25_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

