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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step in plain text first. To avoid malformed output, do not open intermediate << >> spans. Put exactly one final parser-friendly symbolic expression or integer answer inside visible << and >> delimiters at the end, e.g. Final answer: <<expression>>. Preserve variable names exactly, including underscores. For totals include all additive terms; for leftovers subtract every used amount. For percentage questions use int((part)/(whole) * 100). For count/trip questions use // integer division when the answer must be a whole count.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openedAny_: bool
        d_2_openedAny_ = insideConstrained
        d_3_finalArmed_: bool
        d_3_finalArmed_ = insideConstrained
        d_4_forceOpenNext_: bool
        d_4_forceOpenNext_ = False
        d_5_stopAfterClose_: bool
        d_5_stopAfterClose_ = True
        d_6_forceFinalAfter_: int
        d_6_forceFinalAfter_ = 190
        d_7_outsideStopAfter_: int
        d_7_outsideStopAfter_ = 360
        d_8_narrowThreshold_: int
        d_8_narrowThreshold_ = 64
        d_9_penaltyTokens_: _dafny.Seq
        d_9_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " !")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " !!!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "```")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " >>>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " >>>>"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_4_forceOpenNext_) or ((not(d_2_openedAny_)) and ((d_1_steps_) >= (d_6_forceFinalAfter_))):
                            d_10_openedGenerated_: _dafny.Seq
                            d_11_openedInside_: bool
                            d_12_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_10_openedGenerated_ = out0_
                            d_11_openedInside_ = out1_
                            d_12_openedCurrent_ = out2_
                            generated = d_10_openedGenerated_
                            insideConstrainedOut = d_11_openedInside_
                            currentConstrainedOut = d_12_openedCurrent_
                            d_2_openedAny_ = True
                            d_3_finalArmed_ = True
                            d_4_forceOpenNext_ = False
                            d_5_stopAfterClose_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_2_openedAny_) and ((d_1_steps_) >= (d_7_outsideStopAfter_)):
                            raise _dafny.Break("0")
                        elif True:
                            d_13_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_13_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_13_next_) == (eosToken):
                                if (not(d_2_openedAny_)) and ((d_1_steps_) < (maxSteps)):
                                    d_4_forceOpenNext_ = True
                                    d_3_finalArmed_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif ((((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " >>"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>>>"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " >>>>")))):
                                pass
                            elif (((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " <<"))))) and (not(d_3_finalArmed_)):
                                pass
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_13_next_]))
                                if (((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " <<"))))) and (d_3_finalArmed_):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_openedAny_ = True
                                    d_4_forceOpenNext_ = False
                                    d_5_stopAfterClose_ = True
                                elif True:
                                    if ((((((((((((((((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Final"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " answer"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Answer"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " therefore"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Thus"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " thus"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "thus"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "So"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " so")))):
                                        d_3_finalArmed_ = True
                                    if ((((((((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final:")))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final:"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer:"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " answer:"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final answer:"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final answer:"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final answer:"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final answer:")))):
                                        d_3_finalArmed_ = True
                                        d_4_forceOpenNext_ = True
                                    if (d_3_finalArmed_) and (((((((((((((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " :"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ="))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " is"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " are"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " equals"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " is:"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " answer is"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Answer is"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final answer is"))))) or ((d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final answer is"))))):
                                        d_4_forceOpenNext_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_14_closedGenerated_: _dafny.Seq
                        d_15_closedInside_: bool
                        d_16_closedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_14_closedGenerated_ = out4_
                        d_15_closedInside_ = out5_
                        d_16_closedCurrent_ = out6_
                        generated = d_14_closedGenerated_
                        insideConstrainedOut = d_15_closedInside_
                        currentConstrainedOut = d_16_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_5_stopAfterClose_:
                            raise _dafny.Break("0")
                    elif True:
                        d_17_constrainedPrompt_: _dafny.Seq
                        d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_18_nextConstrained_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_9_penaltyTokens_, _dafny.BigRational('5e0'), d_8_narrowThreshold_, eosToken)
                        d_18_nextConstrained_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_18_nextConstrained_) == (eosToken):
                            if (d_1_steps_) < (maxSteps):
                                d_19_candidates_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, 8, eosToken)
                                d_19_candidates_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_20_alt_: _dafny.Seq
                                d_20_alt_ = (d_19_candidates_)[0]
                                if ((d_20_alt_) == (eosToken)) and ((len(d_19_candidates_)) > (1)):
                                    d_20_alt_ = (d_19_candidates_)[1]
                                if ((d_20_alt_) == (eosToken)) and ((len(d_19_candidates_)) > (2)):
                                    d_20_alt_ = (d_19_candidates_)[2]
                                if (d_20_alt_) != (eosToken):
                                    d_21_appendedGenerated_: _dafny.Seq
                                    d_22_appendedInside_: bool
                                    d_23_appendedCurrent_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_alt_)
                                    d_21_appendedGenerated_ = out9_
                                    d_22_appendedInside_ = out10_
                                    d_23_appendedCurrent_ = out11_
                                    generated = d_21_appendedGenerated_
                                    insideConstrainedOut = d_22_appendedInside_
                                    currentConstrainedOut = d_23_appendedCurrent_
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_24_appendedGenerated2_: _dafny.Seq
                            d_25_appendedInside2_: bool
                            d_26_appendedCurrent2_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_nextConstrained_)
                            d_24_appendedGenerated2_ = out12_
                            d_25_appendedInside2_ = out13_
                            d_26_appendedCurrent2_ = out14_
                            generated = d_24_appendedGenerated2_
                            insideConstrainedOut = d_25_appendedInside2_
                            currentConstrainedOut = d_26_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

