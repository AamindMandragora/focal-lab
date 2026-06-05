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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Wrap intermediate symbolic expressions and the final answer inside visible << >> delimiters exactly.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openCount_: int
        d_2_openCount_ = VerifiedDecoderAgent.CSDHelpers.OccurrencesInRange(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), len(generated))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_2_openCount_) == (0):
                            d_3_openedGenerated_: _dafny.Seq
                            d_4_openedInside_: bool
                            d_5_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_3_openedGenerated_ = out0_
                            d_4_openedInside_ = out1_
                            d_5_openedCurrent_ = out2_
                            generated = d_3_openedGenerated_
                            insideConstrainedOut = d_4_openedInside_
                            currentConstrainedOut = d_5_openedCurrent_
                            d_2_openCount_ = VerifiedDecoderAgent.CSDHelpers.OccurrencesInRange(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), len(generated))
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_6_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_6_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_6_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                                if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_openCount_ = VerifiedDecoderAgent.CSDHelpers.OccurrencesInRange(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), len(generated))
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_7_closedGenerated_: _dafny.Seq
                        d_8_closedInside_: bool
                        d_9_closedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_7_closedGenerated_ = out4_
                        d_8_closedInside_ = out5_
                        d_9_closedCurrent_ = out6_
                        generated = d_7_closedGenerated_
                        insideConstrainedOut = d_8_closedInside_
                        currentConstrainedOut = d_9_closedCurrent_
                        d_2_openCount_ = VerifiedDecoderAgent.CSDHelpers.OccurrencesInRange(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), len(generated))
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_10_stablePrefix_: _dafny.Seq
                        d_10_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_11_constrainedPrompt_: _dafny.Seq
                        d_11_constrainedPrompt_ = (prompt) + (d_10_stablePrefix_)
                        d_12_narrow_: bool
                        out7_: bool
                        out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 3)
                        d_12_narrow_ = out7_
                        d_13_next_: _dafny.Seq
                        d_13_next_ = eosToken
                        if d_12_narrow_:
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_13_next_ = out8_
                        elif True:
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_13_next_ = out9_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_13_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_14_appendedGenerated_: _dafny.Seq
                            d_15_appendedInside_: bool
                            d_16_appendedCurrent_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                            d_14_appendedGenerated_ = out10_
                            d_15_appendedInside_ = out11_
                            d_16_appendedCurrent_ = out12_
                            generated = d_14_appendedGenerated_
                            insideConstrainedOut = d_15_appendedInside_
                            currentConstrainedOut = d_16_appendedCurrent_
                            d_2_openCount_ = VerifiedDecoderAgent.CSDHelpers.OccurrencesInRange(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), len(generated))
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

