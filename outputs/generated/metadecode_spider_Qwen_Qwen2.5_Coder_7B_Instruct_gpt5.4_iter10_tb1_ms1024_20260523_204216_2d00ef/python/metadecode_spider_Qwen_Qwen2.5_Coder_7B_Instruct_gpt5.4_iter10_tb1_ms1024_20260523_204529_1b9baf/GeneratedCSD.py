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
        if True:
            generated = generatedPrefix
            insideConstrainedOut = insideConstrained
            currentConstrainedOut = currentConstrained
            cost = 0
            if (maxSteps) == (0):
                cost = 0
            elif (insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                d_1_closedGenerated0_: _dafny.Seq
                d_2_closedInside0_: bool
                d_3_closedCurrent0_: _dafny.Seq
                out0_: _dafny.Seq
                out1_: bool
                out2_: _dafny.Seq
                out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_1_closedGenerated0_ = out0_
                d_2_closedInside0_ = out1_
                d_3_closedCurrent0_ = out2_
                generated = d_1_closedGenerated0_
                insideConstrainedOut = d_2_closedInside0_
                currentConstrainedOut = d_3_closedCurrent0_
                cost = 1
            elif True:
                d_4_stepsUsed_: int
                d_4_stepsUsed_ = 0
                if not(insideConstrainedOut):
                    d_5_openedGenerated_: _dafny.Seq
                    d_6_openedInside_: bool
                    d_7_openedCurrent_: _dafny.Seq
                    out3_: _dafny.Seq
                    out4_: bool
                    out5_: _dafny.Seq
                    out3_, out4_, out5_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_5_openedGenerated_ = out3_
                    d_6_openedInside_ = out4_
                    d_7_openedCurrent_ = out5_
                    generated = d_5_openedGenerated_
                    insideConstrainedOut = d_6_openedInside_
                    currentConstrainedOut = d_7_openedCurrent_
                    d_4_stepsUsed_ = 1
                if ((insideConstrainedOut) and ((d_4_stepsUsed_) < (maxSteps))) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                    d_8_constrainedPrompt_: _dafny.Seq
                    d_8_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_9_next_: _dafny.Seq
                    out6_: _dafny.Seq
                    out6_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                    d_9_next_ = out6_
                    d_4_stepsUsed_ = (d_4_stepsUsed_) + (1)
                    if (d_9_next_) != (eosToken):
                        d_10_appendedGenerated_: _dafny.Seq
                        d_11_appendedInside_: bool
                        d_12_appendedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                        d_10_appendedGenerated_ = out7_
                        d_11_appendedInside_ = out8_
                        d_12_appendedCurrent_ = out9_
                        generated = d_10_appendedGenerated_
                        insideConstrainedOut = d_11_appendedInside_
                        currentConstrainedOut = d_12_appendedCurrent_
                if ((insideConstrainedOut) and ((d_4_stepsUsed_) < (maxSteps))) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                    d_13_closedGenerated_: _dafny.Seq
                    d_14_closedInside_: bool
                    d_15_closedCurrent_: _dafny.Seq
                    out10_: _dafny.Seq
                    out11_: bool
                    out12_: _dafny.Seq
                    out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_13_closedGenerated_ = out10_
                    d_14_closedInside_ = out11_
                    d_15_closedCurrent_ = out12_
                    generated = d_13_closedGenerated_
                    insideConstrainedOut = d_14_closedInside_
                    currentConstrainedOut = d_15_closedCurrent_
                    d_4_stepsUsed_ = (d_4_stepsUsed_) + (1)
                cost = d_4_stepsUsed_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

