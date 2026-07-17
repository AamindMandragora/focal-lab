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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem carefully in plain text first. Put the final algebraic answer after 'Final:' and wrap complete symbolic expressions only in visible << and >> delimiters. Use lowercase variable names exactly as given, Python-style arithmetic, and int(...) for whole-number percentage or count answers when needed.")))
        d_1_effectiveMax_: int
        if (maxSteps) < (320):
            d_1_effectiveMax_ = maxSteps
        elif True:
            d_1_effectiveMax_ = 320
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_finalArmed_: bool
        d_3_finalArmed_ = False
        d_4_forceOpenNext_: bool
        d_4_forceOpenNext_ = False
        d_5_openedByStrategy_: bool
        d_5_openedByStrategy_ = False
        d_6_closedSpans_: int
        d_6_closedSpans_ = 0
        d_7_penaltyTokens_: _dafny.Seq
        d_7_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "{")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "}")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " <<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " >>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "```"))])
        with _dafny.label("0"):
            while (d_2_steps_) < (d_1_effectiveMax_):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_4_forceOpenNext_:
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
                            d_5_openedByStrategy_ = True
                            d_4_forceOpenNext_ = False
                            d_3_finalArmed_ = True
                            d_2_steps_ = (d_2_steps_) + (1)
                        elif True:
                            d_11_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_11_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_11_next_) == (eosToken):
                                if ((d_6_closedSpans_) == (0)) and ((d_2_steps_) < (d_1_effectiveMax_)):
                                    d_4_forceOpenNext_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif (d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                pass
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                if (d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_12_enteredGenerated_: _dafny.Seq
                                    d_13_enteredInside_: bool
                                    d_14_enteredCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_12_enteredGenerated_ = out4_
                                    d_13_enteredInside_ = out5_
                                    d_14_enteredCurrent_ = out6_
                                    generated = d_12_enteredGenerated_
                                    insideConstrainedOut = d_13_enteredInside_
                                    currentConstrainedOut = d_14_enteredCurrent_
                                    d_5_openedByStrategy_ = False
                                    d_4_forceOpenNext_ = False
                                elif True:
                                    if ((((((((((((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " answer"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Answer"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Final"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "result"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " result"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Result"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Result")))):
                                        d_3_finalArmed_ = True
                                    if ((d_3_finalArmed_) and ((d_2_steps_) >= (12))) and ((((((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " :"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " is"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ="))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))))):
                                        d_4_forceOpenNext_ = True
                                    elif ((d_6_closedSpans_) == (0)) and ((d_2_steps_) >= (180)):
                                        d_4_forceOpenNext_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_15_closedGenerated_: _dafny.Seq
                        d_16_closedInside_: bool
                        d_17_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_15_closedGenerated_ = out7_
                        d_16_closedInside_ = out8_
                        d_17_closedCurrent_ = out9_
                        generated = d_15_closedGenerated_
                        insideConstrainedOut = d_16_closedInside_
                        currentConstrainedOut = d_17_closedCurrent_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_6_closedSpans_ = (d_6_closedSpans_) + (1)
                        if ((d_5_openedByStrategy_) or (d_3_finalArmed_)) or ((d_6_closedSpans_) >= (3)):
                            raise _dafny.Break("0")
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_nextConstrained_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_7_penaltyTokens_, _dafny.BigRational('5e0'), 64, eosToken)
                        d_19_nextConstrained_ = out10_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_19_nextConstrained_) == (eosToken):
                            if (d_2_steps_) < (d_1_effectiveMax_):
                                d_20_candidates_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, 4, eosToken)
                                d_20_candidates_ = out11_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_21_alt_: _dafny.Seq
                                d_21_alt_ = (d_20_candidates_)[0]
                                if ((d_21_alt_) == (eosToken)) and ((len(d_20_candidates_)) > (1)):
                                    d_21_alt_ = (d_20_candidates_)[1]
                                if (d_21_alt_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_22_appendedGenerated_: _dafny.Seq
                                    d_23_appendedInside_: bool
                                    d_24_appendedCurrent_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_alt_)
                                    d_22_appendedGenerated_ = out12_
                                    d_23_appendedInside_ = out13_
                                    d_24_appendedCurrent_ = out14_
                                    generated = d_22_appendedGenerated_
                                    insideConstrainedOut = d_23_appendedInside_
                                    currentConstrainedOut = d_24_appendedCurrent_
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_25_appendedGenerated2_: _dafny.Seq
                            d_26_appendedInside2_: bool
                            d_27_appendedCurrent2_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_nextConstrained_)
                            d_25_appendedGenerated2_ = out15_
                            d_26_appendedInside2_ = out16_
                            d_27_appendedCurrent2_ = out17_
                            generated = d_25_appendedGenerated2_
                            insideConstrainedOut = d_26_appendedInside2_
                            currentConstrainedOut = d_27_appendedCurrent2_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

