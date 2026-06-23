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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step, and put the final answer inside a visible << >> span.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_openedGenerated_: _dafny.Seq
                        d_3_openedInside_: bool
                        d_4_openedCurrent_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_2_openedGenerated_ = out0_
                        d_3_openedInside_ = out1_
                        d_4_openedCurrent_ = out2_
                        generated = d_2_openedGenerated_
                        insideConstrainedOut = d_3_openedInside_
                        currentConstrainedOut = d_4_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_5_closedGenerated_: _dafny.Seq
                        d_6_closedInside_: bool
                        d_7_closedCurrent_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_5_closedGenerated_ = out3_
                        d_6_closedInside_ = out4_
                        d_7_closedCurrent_ = out5_
                        generated = d_5_closedGenerated_
                        insideConstrainedOut = d_6_closedInside_
                        currentConstrainedOut = d_7_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_8_stablePrefix_: _dafny.Seq
                        d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_9_constrainedPrompt_: _dafny.Seq
                        d_9_constrainedPrompt_ = (prompt) + (d_8_stablePrefix_)
                        d_10_next_: _dafny.Seq
                        d_10_next_ = eosToken
                        d_11_narrow_: bool
                        out6_: bool
                        out6_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 8)
                        d_11_narrow_ = out6_
                        if d_11_narrow_:
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), generated, _dafny.BigRational('2e0'), 12, eosToken)
                            d_10_next_ = out7_
                        elif True:
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_10_next_ = out8_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_10_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_12_appendedGenerated_: _dafny.Seq
                            d_13_appendedInside_: bool
                            d_14_appendedCurrent_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                            d_12_appendedGenerated_ = out9_
                            d_13_appendedInside_ = out10_
                            d_14_appendedCurrent_ = out11_
                            generated = d_12_appendedGenerated_
                            insideConstrainedOut = d_13_appendedInside_
                            currentConstrainedOut = d_14_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

