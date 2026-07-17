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
                pass
            elif True:
                d_1_steps_: int
                d_1_steps_ = 0
                if ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
                    d_2_openGenerated_: _dafny.Seq
                    d_3_openInside_: bool
                    d_4_openCurrent_: _dafny.Seq
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: _dafny.Seq
                    out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_2_openGenerated_ = out0_
                    d_3_openInside_ = out1_
                    d_4_openCurrent_ = out2_
                    generated = d_2_openGenerated_
                    insideConstrainedOut = d_3_openInside_
                    currentConstrainedOut = d_4_openCurrent_
                    d_1_steps_ = (d_1_steps_) + (1)
                with _dafny.label("1_0"):
                    while (((d_1_steps_) < (maxSteps)) and (insideConstrainedOut)) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                        with _dafny.c_label("1_0"):
                            d_5_constrainedPrompt_: _dafny.Seq
                            d_5_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_6_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_5_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_6_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_6_next_) == (eosToken):
                                raise _dafny.Break("1_0")
                            elif True:
                                d_7_appendedGenerated_: _dafny.Seq
                                d_8_appendedInside_: bool
                                d_9_appendedCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_6_next_)
                                d_7_appendedGenerated_ = out4_
                                d_8_appendedInside_ = out5_
                                d_9_appendedCurrent_ = out6_
                                generated = d_7_appendedGenerated_
                                insideConstrainedOut = d_8_appendedInside_
                                currentConstrainedOut = d_9_appendedCurrent_
                            pass
                    pass
                if (((d_1_steps_) < (maxSteps)) and (insideConstrainedOut)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                    d_10_closedGenerated_: _dafny.Seq
                    d_11_closedInside_: bool
                    d_12_closedCurrent_: _dafny.Seq
                    out7_: _dafny.Seq
                    out8_: bool
                    out9_: _dafny.Seq
                    out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_10_closedGenerated_ = out7_
                    d_11_closedInside_ = out8_
                    d_12_closedCurrent_ = out9_
                    generated = d_10_closedGenerated_
                    insideConstrainedOut = d_11_closedInside_
                    currentConstrainedOut = d_12_closedCurrent_
                    d_1_steps_ = (d_1_steps_) + (1)
                cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

