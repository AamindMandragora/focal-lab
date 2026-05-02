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
        d_2_preferOpen_: bool
        d_2_preferOpen_ = False
        if not(insideConstrainedOut):
            if (len(generated)) == (0):
                d_2_preferOpen_ = True
            elif True:
                d_2_preferOpen_ = True
        while (d_1_steps_) < (maxSteps):
            if not(insideConstrainedOut):
                if d_2_preferOpen_:
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
                    d_2_preferOpen_ = False
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    d_6_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_6_next_ = out3_
                    if (d_6_next_) == (eosToken):
                        d_1_steps_ = maxSteps
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                        d_1_steps_ = (d_1_steps_) + (1)
                        if ((((((d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")))) or ((d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ","))))) or ((d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":"))))) or ((d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))))) or ((d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "is"))))) or ((d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "are")))):
                            d_2_preferOpen_ = True
                        elif True:
                            if (((((((VerifiedDecoderAgent.default__.Contains(d_6_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "step")))) or (VerifiedDecoderAgent.default__.Contains(d_6_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "first"))))) or (VerifiedDecoderAgent.default__.Contains(d_6_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "then"))))) or (VerifiedDecoderAgent.default__.Contains(d_6_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "next"))))) or (VerifiedDecoderAgent.default__.Contains(d_6_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "so"))))) or (VerifiedDecoderAgent.default__.Contains(d_6_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore"))))) or (VerifiedDecoderAgent.default__.Contains(d_6_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "calculate"))))) or (VerifiedDecoderAgent.default__.Contains(d_6_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "compute")))):
                                d_2_preferOpen_ = True
                            elif True:
                                d_2_preferOpen_ = False
            elif True:
                d_7_complete_: bool
                d_7_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                if d_7_complete_:
                    d_8_closedGenerated_: _dafny.Seq
                    d_9_closedInside_: bool
                    d_10_closedCurrent_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_8_closedGenerated_ = out4_
                    d_9_closedInside_ = out5_
                    d_10_closedCurrent_ = out6_
                    generated = d_8_closedGenerated_
                    insideConstrainedOut = d_9_closedInside_
                    currentConstrainedOut = d_10_closedCurrent_
                    d_2_preferOpen_ = False
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    d_11_constrainedPrompt_: _dafny.Seq
                    d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_12_next_: _dafny.Seq
                    out7_: _dafny.Seq
                    out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                    d_12_next_ = out7_
                    if (d_12_next_) == (eosToken):
                        d_1_steps_ = maxSteps
                    elif True:
                        d_13_appendedGenerated_: _dafny.Seq
                        d_14_appendedInside_: bool
                        d_15_appendedCurrent_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                        d_13_appendedGenerated_ = out8_
                        d_14_appendedInside_ = out9_
                        d_15_appendedCurrent_ = out10_
                        generated = d_13_appendedGenerated_
                        insideConstrainedOut = d_14_appendedInside_
                        currentConstrainedOut = d_15_appendedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

