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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Wrap each symbolic calculation and the final answer in visible << and >> delimiters. End with a final line like Final answer: <<expression>>. The extracted expression must be the complete requested quantity, not an intermediate subtotal. Preserve variable names and underscores. For totals include all terms; for leftovers subtract all used amounts; for percentages use int(100 * numerator / denominator) or int((part)/(whole) * 100); for indivisible counts use // integer division when appropriate.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openedAny_: bool
        d_2_openedAny_ = insideConstrained
        d_3_finalArmed_: bool
        d_3_finalArmed_ = False
        d_4_forceOpenNext_: bool
        d_4_forceOpenNext_ = False
        d_5_stopAfterClose_: bool
        d_5_stopAfterClose_ = False
        d_6_noSpanOpenAfter_: int
        d_6_noSpanOpenAfter_ = 120
        d_7_afterSpanStopAfter_: int
        d_7_afterSpanStopAfter_ = 320
        d_8_narrowThreshold_: int
        d_8_narrowThreshold_ = 48
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_4_forceOpenNext_) or ((not(d_2_openedAny_)) and ((d_1_steps_) >= (d_6_noSpanOpenAfter_))):
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
                            d_4_forceOpenNext_ = False
                            d_5_stopAfterClose_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_2_openedAny_) and ((d_1_steps_) >= (d_7_afterSpanStopAfter_)):
                            raise _dafny.Break("0")
                        elif True:
                            d_12_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_12_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_12_next_) == (eosToken):
                                if (not(d_2_openedAny_)) and ((d_1_steps_) < (maxSteps)):
                                    d_4_forceOpenNext_ = True
                                    d_5_stopAfterClose_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " >>")))):
                                pass
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                                if ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " <<")))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_openedAny_ = True
                                    if d_3_finalArmed_:
                                        d_5_stopAfterClose_ = True
                                elif True:
                                    d_13_sawFinalMarker_: bool
                                    d_13_sawFinalMarker_ = False
                                    if ((((((((((((((((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Final"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " answer"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Answer"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final answer"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final answer"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final answer"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Final answer"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer is"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " answer is"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer is"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Answer is")))):
                                        d_13_sawFinalMarker_ = True
                                        d_3_finalArmed_ = True
                                    if ((((((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final:")))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final:"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer:"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " answer:"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final answer:"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " final answer:")))):
                                        d_13_sawFinalMarker_ = True
                                        d_3_finalArmed_ = True
                                        d_4_forceOpenNext_ = True
                                        d_5_stopAfterClose_ = True
                                    if (d_3_finalArmed_) and (((((((((((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " :"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ="))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " is"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " are"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " equals"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " is:"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " answer is"))))) or ((d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Answer is"))))):
                                        d_4_forceOpenNext_ = True
                                        d_5_stopAfterClose_ = True
                                    if (d_13_sawFinalMarker_) and ((d_1_steps_) < (maxSteps)):
                                        d_3_finalArmed_ = True
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
                        out7_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_8_narrowThreshold_, eosToken)
                        d_18_nextConstrained_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_18_nextConstrained_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_19_appendedGenerated_: _dafny.Seq
                            d_20_appendedInside_: bool
                            d_21_appendedCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_nextConstrained_)
                            d_19_appendedGenerated_ = out8_
                            d_20_appendedInside_ = out9_
                            d_21_appendedCurrent_ = out10_
                            generated = d_19_appendedGenerated_
                            insideConstrainedOut = d_20_appendedInside_
                            currentConstrainedOut = d_21_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

