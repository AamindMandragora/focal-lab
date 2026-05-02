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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, eosToken):
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
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openedOnce_: bool
        d_2_openedOnce_ = insideConstrained
        d_3_closedOnce_: bool
        d_3_closedOnce_ = False
        d_4_rationaleBudget_: int
        d_4_rationaleBudget_ = 12
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (not(d_3_closedOnce_)):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_2_openedOnce_):
                            if (d_1_steps_) < (d_4_rationaleBudget_):
                                (lm).GenerateLogits((prompt) + (generated))
                                (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('1e2'))
                                d_5_next_: _dafny.Seq
                                out0_: _dafny.Seq
                                out0_ = (lm).ChooseNextTokenUnconstrained()
                                d_5_next_ = out0_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_5_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                            elif True:
                                d_6_openedGenerated_: _dafny.Seq
                                d_7_openedInside_: bool
                                d_8_openedCurrent_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_6_openedGenerated_ = out1_
                                d_7_openedInside_ = out2_
                                d_8_openedCurrent_ = out3_
                                generated = d_6_openedGenerated_
                                insideConstrainedOut = d_7_openedInside_
                                currentConstrainedOut = d_8_openedCurrent_
                                d_2_openedOnce_ = True
                                d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        d_9_isComplete_: bool
                        d_9_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_9_isComplete_:
                            if ((d_1_steps_) + (1)) <= (maxSteps):
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
                                d_3_closedOnce_ = True
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            if ((d_1_steps_) + (1)) >= (maxSteps):
                                raise _dafny.Break("0")
                            elif True:
                                d_13_constrainedPrompt_: _dafny.Seq
                                d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_14_next2_: _dafny.Seq
                                out7_: _dafny.Seq
                                out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_14_next2_ = out7_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_14_next2_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_15_appendedGenerated_: _dafny.Seq
                                    d_16_appendedInside_: bool
                                    d_17_appendedCurrent_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out9_: bool
                                    out10_: _dafny.Seq
                                    out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next2_)
                                    d_15_appendedGenerated_ = out8_
                                    d_16_appendedInside_ = out9_
                                    d_17_appendedCurrent_ = out10_
                                    generated = d_15_appendedGenerated_
                                    insideConstrainedOut = d_16_appendedInside_
                                    currentConstrainedOut = d_17_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

