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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step before the final answer. Use visible <<expression>> spans for symbolic calculations and the final answer. Do not open << until the expression is ready. Preserve variable names exactly, including underscores like n_1 and n_2. Use // for whole-number counts, trips, remaining groups, or divisible-rate questions when appropriate. Use int((part)/(whole) * 100) for integer percentages. Keep final expressions short and plain, not LaTeX.")))
        d_1_hardLimit_: int
        if (maxSteps) < (420):
            d_1_hardLimit_ = maxSteps
        elif True:
            d_1_hardLimit_ = 420
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_openedAny_: bool
        d_3_openedAny_ = insideConstrained
        d_4_finalArmed_: bool
        d_4_finalArmed_ = False
        d_5_forceOpenNext_: bool
        d_5_forceOpenNext_ = False
        d_6_stopAfterClose_: bool
        d_6_stopAfterClose_ = False
        d_7_penaltyTokens_: _dafny.Seq
        d_7_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\[")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\\]")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " !=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " if")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "if")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " then")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "then")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " else")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "else")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " <<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " >>"))])
        with _dafny.label("0"):
            while (d_2_steps_) < (d_1_hardLimit_):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_3_openedAny_) and ((d_2_steps_) >= (320)):
                            raise _dafny.Break("0")
                        elif ((d_5_forceOpenNext_) or ((not(d_3_openedAny_)) and ((d_2_steps_) >= (140)))) and (not(d_3_openedAny_)):
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
                            d_3_openedAny_ = True
                            d_5_forceOpenNext_ = False
                            d_6_stopAfterClose_ = True
                            d_2_steps_ = (d_2_steps_) + (1)
                        elif True:
                            d_11_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_11_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_11_next_) == (eosToken):
                                if (not(d_3_openedAny_)) and ((d_2_steps_) < (d_1_hardLimit_)):
                                    d_5_forceOpenNext_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " >>")))):
                                pass
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                if ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " <<")))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_3_openedAny_ = True
                                    if d_4_finalArmed_:
                                        d_6_stopAfterClose_ = True
                                elif True:
                                    if ((((((((((((((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Final"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " answer"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Answer"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " therefore"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Thus"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " thus"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "thus")))):
                                        d_4_finalArmed_ = True
                                    if (((not(d_3_openedAny_)) and (d_4_finalArmed_)) and ((d_2_steps_) >= (24))) and ((((((((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " :"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ="))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " is"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " are"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " equals"))))):
                                        d_5_forceOpenNext_ = True
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
                        d_2_steps_ = (d_2_steps_) + (1)
                        if d_6_stopAfterClose_:
                            raise _dafny.Break("0")
                    elif (((d_2_steps_) + (1)) == (d_1_hardLimit_)) or ((len(currentConstrainedOut)) >= (48)):
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                        insideConstrainedOut = False
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_16_nextConstrained_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_7_penaltyTokens_, _dafny.BigRational('2e0'), 36, eosToken)
                        d_16_nextConstrained_ = out7_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_16_nextConstrained_) == (eosToken):
                            if ((d_2_steps_) + (1)) < (d_1_hardLimit_):
                                d_17_candidates_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, 6, eosToken)
                                d_17_candidates_ = out8_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_18_alt_: _dafny.Seq
                                d_18_alt_ = (d_17_candidates_)[0]
                                if ((d_18_alt_) == (eosToken)) and ((len(d_17_candidates_)) > (1)):
                                    d_18_alt_ = (d_17_candidates_)[1]
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
                                elif ((len(currentConstrainedOut)) > (0)) and ((d_2_steps_) < (d_1_hardLimit_)):
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    raise _dafny.Break("0")
                            elif ((len(currentConstrainedOut)) > (0)) and ((d_2_steps_) < (d_1_hardLimit_)):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_steps_ = (d_2_steps_) + (1)
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
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

