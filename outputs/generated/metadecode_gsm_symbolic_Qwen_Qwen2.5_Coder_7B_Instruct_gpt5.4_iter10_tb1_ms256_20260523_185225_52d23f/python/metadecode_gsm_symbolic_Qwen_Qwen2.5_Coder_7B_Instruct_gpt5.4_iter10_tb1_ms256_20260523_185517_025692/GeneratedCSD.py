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
            (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Write each arithmetic computation inside << >> delimiters.")))
            d_1_steps_: int
            d_1_steps_ = 0
            d_2_openedAny_: bool
            d_2_openedAny_ = insideConstrained
            with _dafny.label("0"):
                while (d_1_steps_) < (maxSteps):
                    with _dafny.c_label("0"):
                        if (not(insideConstrainedOut)) and (not(d_2_openedAny_)):
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
                            d_2_openedAny_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif not(insideConstrainedOut):
                            d_6_nextOutside_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_6_nextOutside_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_6_nextOutside_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_nextOutside_]))
                                if (d_6_nextOutside_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_7_enteredGenerated_: _dafny.Seq
                                    d_8_enteredInside_: bool
                                    d_9_enteredCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_7_enteredGenerated_ = out4_
                                    d_8_enteredInside_ = out5_
                                    d_9_enteredCurrent_ = out6_
                                    generated = d_7_enteredGenerated_
                                    insideConstrainedOut = d_8_enteredInside_
                                    currentConstrainedOut = d_9_enteredCurrent_
                                    d_2_openedAny_ = True
                        elif (parser).IsCompletePrefix(currentConstrainedOut):
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
                        elif True:
                            d_13_constrainedPrompt_: _dafny.Seq
                            d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_14_nextInside_: _dafny.Seq
                            d_15_wasConstrained_: bool
                            out10_: _dafny.Seq
                            out11_: bool
                            out10_, out11_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_14_nextInside_ = out10_
                            d_15_wasConstrained_ = out11_
                            if d_15_wasConstrained_:
                                d_16_nextSoft_: _dafny.Seq
                                d_17_usedFallback_: bool
                                out12_: _dafny.Seq
                                out13_: bool
                                out12_, out13_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                                d_16_nextSoft_ = out12_
                                d_17_usedFallback_ = out13_
                                d_14_nextInside_ = d_16_nextSoft_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_14_nextInside_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_18_appendedGenerated_: _dafny.Seq
                                d_19_appendedInside_: bool
                                d_20_appendedCurrent_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_nextInside_)
                                d_18_appendedGenerated_ = out14_
                                d_19_appendedInside_ = out15_
                                d_20_appendedCurrent_ = out16_
                                generated = d_18_appendedGenerated_
                                insideConstrainedOut = d_19_appendedInside_
                                currentConstrainedOut = d_20_appendedCurrent_
                        pass
                pass
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

