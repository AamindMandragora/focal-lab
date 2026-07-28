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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Wrap every intermediate arithmetic expression and the final numerical answer inside << >> delimiters. Example: She bought <<3 * 4 = 12>> apples and <<12 - 5 = 7>> are left. Final answer: <<7>>.")))
        d_1_mathDigitGroup_: _dafny.Seq
        d_1_mathDigitGroup_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9"))])
        d_2_mathOpGroup_: _dafny.Seq
        d_2_mathOpGroup_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ","))])
        d_3_mathGroups_: _dafny.Seq
        d_3_mathGroups_ = (_dafny.SeqWithoutIsStrInference([d_1_mathDigitGroup_, d_2_mathOpGroup_])) + (validTokenGroups)
        d_4_steps_: int
        d_4_steps_ = 0
        d_5_chunkSize_: int
        d_5_chunkSize_ = 40
        with _dafny.label("0"):
            while (d_4_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_budget_: int
                        if ((maxSteps) - (d_4_steps_)) > (d_5_chunkSize_):
                            d_6_budget_ = d_5_chunkSize_
                        elif True:
                            d_6_budget_ = (maxSteps) - (d_4_steps_)
                        d_7_chunkGenerated_: _dafny.Seq
                        d_8_stoppedOnSpan_: bool
                        d_9_stoppedOnEos_: bool
                        d_10_chunkSteps_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_budget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_7_chunkGenerated_ = out0_
                        d_8_stoppedOnSpan_ = out1_
                        d_9_stoppedOnEos_ = out2_
                        d_10_chunkSteps_ = out3_
                        generated = d_7_chunkGenerated_
                        d_4_steps_ = (d_4_steps_) + (d_10_chunkSteps_)
                        if d_9_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_8_stoppedOnSpan_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            if (d_4_steps_) < (maxSteps):
                                d_11_openGenerated_: _dafny.Seq
                                d_12_openInside_: bool
                                d_13_openCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_11_openGenerated_ = out4_
                                d_12_openInside_ = out5_
                                d_13_openCurrent_ = out6_
                                generated = d_11_openGenerated_
                                insideConstrainedOut = d_12_openInside_
                                currentConstrainedOut = d_13_openCurrent_
                                d_4_steps_ = (d_4_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_14_closedGenerated_: _dafny.Seq
                        d_15_closedInside_: bool
                        d_16_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_14_closedGenerated_ = out7_
                        d_15_closedInside_ = out8_
                        d_16_closedCurrent_ = out9_
                        generated = d_14_closedGenerated_
                        insideConstrainedOut = d_15_closedInside_
                        currentConstrainedOut = d_16_closedCurrent_
                        d_4_steps_ = (d_4_steps_) + (1)
                    elif True:
                        d_17_constrainedPrompt_: _dafny.Seq
                        d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_18_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, d_3_mathGroups_, _dafny.BigRational('4e0'), 12, eosToken)
                        d_18_next_ = out10_
                        d_4_steps_ = (d_4_steps_) + (1)
                        if (d_18_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_19_appendedGenerated_: _dafny.Seq
                            d_20_appendedInside_: bool
                            d_21_appendedCurrent_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                            d_19_appendedGenerated_ = out11_
                            d_20_appendedInside_ = out12_
                            d_21_appendedCurrent_ = out13_
                            generated = d_19_appendedGenerated_
                            insideConstrainedOut = d_20_appendedInside_
                            currentConstrainedOut = d_21_appendedCurrent_
                    pass
            pass
        cost = d_4_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

