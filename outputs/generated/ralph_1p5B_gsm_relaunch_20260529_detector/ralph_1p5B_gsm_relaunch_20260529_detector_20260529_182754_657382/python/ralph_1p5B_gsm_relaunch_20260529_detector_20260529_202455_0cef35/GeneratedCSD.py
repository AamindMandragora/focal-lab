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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve math word problems step by step. For EVERY arithmetic expression and the final answer, wrap it in << >> delimiters. Example: <<3 + 4>> = <<7>>. Do NOT emit >> outside of << >> spans.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_unconstrainedTokensSinceLastSpan_: int
        d_2_unconstrainedTokensSinceLastSpan_ = 0
        d_3_forceSpanThreshold_: int
        d_3_forceSpanThreshold_ = 55
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_2_unconstrainedTokensSinceLastSpan_) >= (d_3_forceSpanThreshold_)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                            d_4_openGenerated_: _dafny.Seq
                            d_5_openInside_: bool
                            d_6_openCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_openGenerated_ = out0_
                            d_5_openInside_ = out1_
                            d_6_openCurrent_ = out2_
                            generated = d_4_openGenerated_
                            insideConstrainedOut = d_5_openInside_
                            currentConstrainedOut = d_6_openCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_unconstrainedTokensSinceLastSpan_ = 0
                        elif True:
                            d_7_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_7_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                pass
                            elif (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_unconstrainedTokensSinceLastSpan_ = 0
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                                d_2_unconstrainedTokensSinceLastSpan_ = (d_2_unconstrainedTokensSinceLastSpan_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        if ((len(currentConstrainedOut)) > (0)) and (((currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))):
                            d_8_rolledGenerated_: _dafny.Seq
                            d_9_rolledCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: _dafny.Seq
                            out4_, out5_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_8_rolledGenerated_ = out4_
                            d_9_rolledCurrent_ = out5_
                            generated = d_8_rolledGenerated_
                            currentConstrainedOut = d_9_rolledCurrent_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                d_10_closedGenerated_: _dafny.Seq
                                d_11_closedInside_: bool
                                d_12_closedCurrent_: _dafny.Seq
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: _dafny.Seq
                                out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_10_closedGenerated_ = out6_
                                d_11_closedInside_ = out7_
                                d_12_closedCurrent_ = out8_
                                generated = d_10_closedGenerated_
                                insideConstrainedOut = d_11_closedInside_
                                currentConstrainedOut = d_12_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_unconstrainedTokensSinceLastSpan_ = 0
                            elif ((d_1_steps_) + (1)) <= (maxSteps):
                                d_13_constrainedPrompt_: _dafny.Seq
                                d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_14_next_: _dafny.Seq
                                d_15_wasConstrained_: bool
                                out9_: _dafny.Seq
                                out10_: bool
                                out9_, out10_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_14_next_ = out9_
                                d_15_wasConstrained_ = out10_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_14_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif (d_14_next_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                    d_16_appendedGenerated_: _dafny.Seq
                                    d_17_appendedInside_: bool
                                    d_18_appendedCurrent_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                    d_16_appendedGenerated_ = out11_
                                    d_17_appendedInside_ = out12_
                                    d_18_appendedCurrent_ = out13_
                                    generated = d_16_appendedGenerated_
                                    insideConstrainedOut = d_17_appendedInside_
                                    currentConstrainedOut = d_18_appendedCurrent_
                            elif True:
                                raise _dafny.Break("0")
                        elif ((d_1_steps_) + (1)) <= (maxSteps):
                            d_19_closedGenerated_: _dafny.Seq
                            d_20_closedInside_: bool
                            d_21_closedCurrent_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_19_closedGenerated_ = out14_
                            d_20_closedInside_ = out15_
                            d_21_closedCurrent_ = out16_
                            generated = d_19_closedGenerated_
                            insideConstrainedOut = d_20_closedInside_
                            currentConstrainedOut = d_21_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_unconstrainedTokensSinceLastSpan_ = 0
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        d_22_constrainedPrompt_: _dafny.Seq
                        d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_23_next_: _dafny.Seq
                        d_24_wasConstrained_: bool
                        out17_: _dafny.Seq
                        out18_: bool
                        out17_, out18_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_23_next_ = out17_
                        d_24_wasConstrained_ = out18_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_23_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_25_appendedGenerated_: _dafny.Seq
                            d_26_appendedInside_: bool
                            d_27_appendedCurrent_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: _dafny.Seq
                            out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                            d_25_appendedGenerated_ = out19_
                            d_26_appendedInside_ = out20_
                            d_27_appendedCurrent_ = out21_
                            generated = d_25_appendedGenerated_
                            insideConstrainedOut = d_26_appendedInside_
                            currentConstrainedOut = d_27_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

