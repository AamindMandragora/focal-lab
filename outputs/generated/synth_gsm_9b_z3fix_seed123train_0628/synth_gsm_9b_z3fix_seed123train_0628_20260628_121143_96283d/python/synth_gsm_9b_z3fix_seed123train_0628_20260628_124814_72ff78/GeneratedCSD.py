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
        out0_: _dafny.Seq
        out1_: bool
        out2_: _dafny.Seq
        out3_: int
        out0_, out1_, out2_, out3_ = default__.AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken)
        generated = out0_
        insideConstrainedOut = out1_
        currentConstrainedOut = out2_
        cost = out3_
        if ((maxSteps) > (0)) and ((cost) <= (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

    @staticmethod
    def AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. At the end, write the final symbolic answer expression inside << >> delimiters. Use only variable names from the problem (no curly braces, no ** operator). Example: <<n * m / k>>"))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_reserveForSpan_: int
                        d_3_reserveForSpan_ = 30
                        if ((d_2_steps_) + (d_3_reserveForSpan_)) >= (maxSteps):
                            d_4_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_4_next_ = out0_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_4_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                                if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            d_5_chunkSize_: int
                            d_5_chunkSize_ = 20
                            if ((d_2_steps_) + (d_5_chunkSize_)) > ((maxSteps) - (d_3_reserveForSpan_)):
                                d_5_chunkSize_ = ((maxSteps) - (d_3_reserveForSpan_)) - (d_2_steps_)
                            if (d_5_chunkSize_) == (0):
                                d_6_next_: _dafny.Seq
                                out1_: _dafny.Seq
                                out1_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                d_6_next_ = out1_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if (d_6_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                                    if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                        insideConstrainedOut = True
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif True:
                                d_7_genOut_: _dafny.Seq
                                d_8_stoppedOnOpen_: bool
                                d_9_stoppedOnEos_: bool
                                d_10_stepsUsed_: int
                                out2_: _dafny.Seq
                                out3_: bool
                                out4_: bool
                                out5_: int
                                out2_, out3_, out4_, out5_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                                d_7_genOut_ = out2_
                                d_8_stoppedOnOpen_ = out3_
                                d_9_stoppedOnEos_ = out4_
                                d_10_stepsUsed_ = out5_
                                d_2_steps_ = (d_2_steps_) + (d_10_stepsUsed_)
                                generated = d_7_genOut_
                                if d_9_stoppedOnEos_:
                                    raise _dafny.Break("0")
                                elif d_8_stoppedOnOpen_:
                                    out6_: _dafny.Seq
                                    out7_: bool
                                    out8_: _dafny.Seq
                                    out6_, out7_, out8_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    generated = out6_
                                    insideConstrainedOut = out7_
                                    currentConstrainedOut = out8_
                                elif True:
                                    if ((len(generated)) > (0)) and (((generated)[(len(generated)) - (1)]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))):
                                        out9_: _dafny.Seq
                                        out10_: bool
                                        out11_: _dafny.Seq
                                        out9_, out10_, out11_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                        generated = out9_
                                        insideConstrainedOut = out10_
                                        currentConstrainedOut = out11_
                    elif True:
                        d_11_cg_: _dafny.Seq
                        d_12_ci_: bool
                        d_13_cc_: _dafny.Seq
                        d_14_closed_: bool
                        out12_: _dafny.Seq
                        out13_: bool
                        out14_: _dafny.Seq
                        out15_: bool
                        out12_, out13_, out14_, out15_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_11_cg_ = out12_
                        d_12_ci_ = out13_
                        d_13_cc_ = out14_
                        d_14_closed_ = out15_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if d_14_closed_:
                            generated = d_11_cg_
                            insideConstrainedOut = d_12_ci_
                            currentConstrainedOut = d_13_cc_
                        elif True:
                            if ((d_2_steps_) + (5)) >= (maxSteps):
                                d_15_closeBudget_: int
                                d_15_closeBudget_ = (maxSteps) - (d_2_steps_)
                                if (d_15_closeBudget_) >= (1):
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_15_closeBudget_)
                                    generated = out16_
                                    insideConstrainedOut = out17_
                                    currentConstrainedOut = out18_
                                    d_2_steps_ = maxSteps
                            elif True:
                                d_16_constrainedPrompt_: _dafny.Seq
                                d_16_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_17_next_: _dafny.Seq
                                out19_: _dafny.Seq
                                out19_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 15, eosToken)
                                d_17_next_ = out19_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if (d_17_next_) == (eosToken):
                                    d_18_closeBudget2_: int
                                    d_18_closeBudget2_ = (maxSteps) - (d_2_steps_)
                                    if (d_18_closeBudget2_) >= (1):
                                        out20_: _dafny.Seq
                                        out21_: bool
                                        out22_: _dafny.Seq
                                        out20_, out21_, out22_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_18_closeBudget2_)
                                        generated = out20_
                                        insideConstrainedOut = out21_
                                        currentConstrainedOut = out22_
                                        d_2_steps_ = maxSteps
                                    raise _dafny.Break("0")
                                elif True:
                                    d_19_appendedGenerated_: _dafny.Seq
                                    d_20_appendedInside_: bool
                                    d_21_appendedCurrent_: _dafny.Seq
                                    out23_: _dafny.Seq
                                    out24_: bool
                                    out25_: _dafny.Seq
                                    out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                    d_19_appendedGenerated_ = out23_
                                    d_20_appendedInside_ = out24_
                                    d_21_appendedCurrent_ = out25_
                                    generated = d_19_appendedGenerated_
                                    insideConstrainedOut = d_20_appendedInside_
                                    currentConstrainedOut = d_21_appendedCurrent_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_22_finalBudget_: int
            d_22_finalBudget_ = (maxSteps) - (d_2_steps_)
            out26_: _dafny.Seq
            out27_: bool
            out28_: _dafny.Seq
            out26_, out27_, out28_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_22_finalBudget_)
            generated = out26_
            insideConstrainedOut = out27_
            currentConstrainedOut = out28_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

