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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Write each arithmetic sub-expression inside << >> delimiters as you go. End with #### <<final_answer>>. Always close every << with >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeTokensSinceLastClose_: int
        d_2_freeTokensSinceLastClose_ = 0
        d_3_totalSpansOpened_: int
        d_3_totalSpansOpened_ = 0
        d_4_maxForcedSpans_: int
        d_4_maxForcedSpans_ = 6
        d_5_spanOpenThreshold_: int
        d_5_spanOpenThreshold_ = 28
        d_6_tokensSinceSpanOpen_: int
        d_6_tokensSinceSpanOpen_ = 0
        d_7_maxTokensInSpan_: int
        d_7_maxTokensInSpan_ = 40
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_8_budgetRemaining_: int
                        d_8_budgetRemaining_ = (maxSteps) - (d_1_steps_)
                        d_9_shouldForce_: bool
                        d_9_shouldForce_ = (((d_2_freeTokensSinceLastClose_) >= (d_5_spanOpenThreshold_)) and ((d_3_totalSpansOpened_) < (d_4_maxForcedSpans_))) and ((d_8_budgetRemaining_) >= (4))
                        if d_9_shouldForce_:
                            d_10_openGenerated_: _dafny.Seq
                            d_11_openInside_: bool
                            d_12_openCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_10_openGenerated_ = out0_
                            d_11_openInside_ = out1_
                            d_12_openCurrent_ = out2_
                            generated = d_10_openGenerated_
                            insideConstrainedOut = d_11_openInside_
                            currentConstrainedOut = d_12_openCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_totalSpansOpened_ = (d_3_totalSpansOpened_) + (1)
                            d_2_freeTokensSinceLastClose_ = 0
                            d_6_tokensSinceSpanOpen_ = 0
                        elif True:
                            d_13_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_13_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_13_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif (d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_13_next_]))
                                d_14_g2_: _dafny.Seq
                                d_15_ins2_: bool
                                d_16_cur2_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_14_g2_ = out4_
                                d_15_ins2_ = out5_
                                d_16_cur2_ = out6_
                                generated = d_14_g2_
                                insideConstrainedOut = d_15_ins2_
                                currentConstrainedOut = d_16_cur2_
                                d_3_totalSpansOpened_ = (d_3_totalSpansOpened_) + (1)
                                d_2_freeTokensSinceLastClose_ = 0
                                d_6_tokensSinceSpanOpen_ = 0
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_13_next_]))
                                d_2_freeTokensSinceLastClose_ = (d_2_freeTokensSinceLastClose_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_17_closedGenerated_: _dafny.Seq
                        d_18_closedInside_: bool
                        d_19_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_17_closedGenerated_ = out7_
                        d_18_closedInside_ = out8_
                        d_19_closedCurrent_ = out9_
                        generated = d_17_closedGenerated_
                        insideConstrainedOut = d_18_closedInside_
                        currentConstrainedOut = d_19_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_freeTokensSinceLastClose_ = 0
                        d_6_tokensSinceSpanOpen_ = 0
                    elif True:
                        d_20_constrainedPrompt_: _dafny.Seq
                        d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_21_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if (d_6_tokensSinceSpanOpen_) >= (d_7_maxTokensInSpan_):
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_21_next_ = out10_
                        elif True:
                            d_22_wasConstrained_: bool = False
                            out11_: _dafny.Seq
                            out12_: bool
                            out11_, out12_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_21_next_ = out11_
                            d_22_wasConstrained_ = out12_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_6_tokensSinceSpanOpen_ = (d_6_tokensSinceSpanOpen_) + (1)
                        if (d_21_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_23_appendedGenerated_: _dafny.Seq
                            d_24_appendedInside_: bool
                            d_25_appendedCurrent_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                            d_23_appendedGenerated_ = out13_
                            d_24_appendedInside_ = out14_
                            d_25_appendedCurrent_ = out15_
                            generated = d_23_appendedGenerated_
                            insideConstrainedOut = d_24_appendedInside_
                            currentConstrainedOut = d_25_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

