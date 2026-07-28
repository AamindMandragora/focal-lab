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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Begin directly with the solution; do not acknowledge these instructions. Reason step by step in plain text, but reserve visible delimiters for the final symbolic answer only: end with exactly one final expression in << and >>. The extracted answer must be the complete requested quantity, not an intermediate subtotal. Preserve variable names and underscores. For totals include all terms; for leftovers subtract all used amounts; for percentages use int(100 * numerator / denominator) or int((part)/(whole) * 100); for indivisible counts use // integer division when appropriate.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openedAny_: bool
        d_2_openedAny_ = insideConstrained
        d_3_finalArmed_: bool
        d_3_finalArmed_ = False
        d_4_forceOpenNext_: bool
        d_4_forceOpenNext_ = False
        d_5_penaltyTokens_: _dafny.Seq
        d_5_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "  ")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "   ")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " unit")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " units")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " cup")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " cups")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " teaspoon")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " teaspoons")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " dollar")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " dollars")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " meters")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " litres")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " liters")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "."))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (((d_4_forceOpenNext_) or (((not(d_2_openedAny_)) and (d_3_finalArmed_)) and ((d_1_steps_) >= (24)))) or ((not(d_2_openedAny_)) and ((d_1_steps_) >= (160)))) and (not(d_2_openedAny_)):
                            d_6_openedGenerated_: _dafny.Seq
                            d_7_openedInside_: bool
                            d_8_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_openedGenerated_ = out0_
                            d_7_openedInside_ = out1_
                            d_8_openedCurrent_ = out2_
                            generated = d_6_openedGenerated_
                            insideConstrainedOut = d_7_openedInside_
                            currentConstrainedOut = d_8_openedCurrent_
                            d_2_openedAny_ = True
                            d_4_forceOpenNext_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_9_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                if (not(d_2_openedAny_)) and ((d_1_steps_) < (maxSteps)):
                                    d_4_forceOpenNext_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " >>")))):
                                pass
                            elif ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " <<")))):
                                if (d_3_finalArmed_) or ((d_1_steps_) >= (80)):
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_openedAny_ = True
                                elif True:
                                    pass
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                if ((((((((((((((((((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final"))))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Final"))))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " answer"))))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Answer"))))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore"))))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " therefore"))))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore"))))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Thus"))))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " thus"))))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "thus"))))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Hence"))))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " hence"))))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "So"))))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " so")))):
                                    d_3_finalArmed_ = True
                                if ((not(d_2_openedAny_)) and (d_3_finalArmed_)) and ((((((((((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " :"))))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ="))))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " is"))))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " are"))))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " equals"))))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " is:"))))) or ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " answer is"))))):
                                    d_4_forceOpenNext_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_10_closedGenerated_: _dafny.Seq
                        d_11_closedInside_: bool
                        d_12_closedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_10_closedGenerated_ = out4_
                        d_11_closedInside_ = out5_
                        d_12_closedCurrent_ = out6_
                        generated = d_10_closedGenerated_
                        insideConstrainedOut = d_11_closedInside_
                        currentConstrainedOut = d_12_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_13_constrainedPrompt_: _dafny.Seq
                        d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_14_nextConstrained_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_5_penaltyTokens_, _dafny.BigRational('5e0'), 48, eosToken)
                        d_14_nextConstrained_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_14_nextConstrained_) == (eosToken):
                            if (d_1_steps_) < (maxSteps):
                                d_15_candidates_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, 6, eosToken)
                                d_15_candidates_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_16_alt_: _dafny.Seq
                                d_16_alt_ = (d_15_candidates_)[0]
                                if ((d_16_alt_) == (eosToken)) and ((len(d_15_candidates_)) > (1)):
                                    d_16_alt_ = (d_15_candidates_)[1]
                                if (d_16_alt_) != (eosToken):
                                    d_17_appendedGenerated_: _dafny.Seq
                                    d_18_appendedInside_: bool
                                    d_19_appendedCurrent_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_alt_)
                                    d_17_appendedGenerated_ = out9_
                                    d_18_appendedInside_ = out10_
                                    d_19_appendedCurrent_ = out11_
                                    generated = d_17_appendedGenerated_
                                    insideConstrainedOut = d_18_appendedInside_
                                    currentConstrainedOut = d_19_appendedCurrent_
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_20_appendedGenerated2_: _dafny.Seq
                            d_21_appendedInside2_: bool
                            d_22_appendedCurrent2_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_nextConstrained_)
                            d_20_appendedGenerated2_ = out12_
                            d_21_appendedInside2_ = out13_
                            d_22_appendedCurrent2_ = out14_
                            generated = d_20_appendedGenerated2_
                            insideConstrainedOut = d_21_appendedInside2_
                            currentConstrainedOut = d_22_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

