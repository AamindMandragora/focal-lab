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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Wrap every intermediate symbolic arithmetic expression and the final answer inside visible << >> delimiters. Inside each span write only a valid arithmetic expression: no prose, no units, no Markdown, no LaTeX, and no placeholder text. Preserve variable names exactly as given, including underscores. Use int(...) for integer percentage or money answers when needed.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeSteps_: int
        d_2_freeSteps_ = 0
        d_3_reasoningLimit_: int
        d_3_reasoningLimit_ = 80
        d_4_reserveForFinal_: int
        d_4_reserveForFinal_ = 70
        d_5_penaltyTokens_: _dafny.Seq
        d_5_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "the")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "The")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " a")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " an")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "of")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "is")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "are")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "and")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "then")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "so")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "expression")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Expression")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "placeholder")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "feet")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "foot")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "inches")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "dollars")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "dollar")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "cents")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "percent")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "percentage")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "units")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "unit")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Markdown")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LaTeX")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "?")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!")), eosToken])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_2_freeSteps_) < (d_3_reasoningLimit_)) and (((d_1_steps_) + (d_4_reserveForFinal_)) < (maxSteps)):
                            d_6_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_6_next_ = out0_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_freeSteps_ = (d_2_freeSteps_) + (1)
                            if (d_6_next_) == (eosToken):
                                d_2_freeSteps_ = d_3_reasoningLimit_
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                                if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_freeSteps_ = d_3_reasoningLimit_
                        elif True:
                            d_7_openedGenerated_: _dafny.Seq
                            d_8_openedInside_: bool
                            d_9_openedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_7_openedGenerated_ = out1_
                            d_8_openedInside_ = out2_
                            d_9_openedCurrent_ = out3_
                            generated = d_7_openedGenerated_
                            insideConstrainedOut = d_8_openedInside_
                            currentConstrainedOut = d_9_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
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
                    elif True:
                        d_13_stablePrefix_: _dafny.Seq
                        d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_14_constrainedPrompt_: _dafny.Seq
                        d_14_constrainedPrompt_ = (prompt) + (d_13_stablePrefix_)
                        d_15_nextPen_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), d_5_penaltyTokens_, _dafny.BigRational('8e0'), 20, eosToken)
                        d_15_nextPen_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_15_nextPen_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_16_appendedGenerated_: _dafny.Seq
                            d_17_appendedInside_: bool
                            d_18_appendedCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_nextPen_)
                            d_16_appendedGenerated_ = out8_
                            d_17_appendedInside_ = out9_
                            d_18_appendedCurrent_ = out10_
                            generated = d_16_appendedGenerated_
                            insideConstrainedOut = d_17_appendedInside_
                            currentConstrainedOut = d_18_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

