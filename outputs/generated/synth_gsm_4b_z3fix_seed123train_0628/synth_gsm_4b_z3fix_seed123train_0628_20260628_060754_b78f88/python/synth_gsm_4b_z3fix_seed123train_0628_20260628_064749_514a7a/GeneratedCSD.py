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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Show reasoning in plain text. Place ONLY the final answer expression inside << >>. Use only the variables from the problem. Do not use curly braces or markdown inside << >>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_closeReserve_: int
        d_3_closeReserve_ = 10
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if (insideConstrainedOut) and (((d_2_steps_) + (d_3_closeReserve_)) >= (maxSteps)):
                        d_4_closeBudget_: int
                        d_4_closeBudget_ = (maxSteps) - (d_2_steps_)
                        if (d_4_closeBudget_) >= (1):
                            d_5_cg_: _dafny.Seq
                            d_6_ci_: bool
                            d_7_cc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_4_closeBudget_)
                            d_5_cg_ = out0_
                            d_6_ci_ = out1_
                            d_7_cc_ = out2_
                            generated = d_5_cg_
                            insideConstrainedOut = d_6_ci_
                            currentConstrainedOut = d_7_cc_
                            d_2_steps_ = maxSteps
                        raise _dafny.Break("0")
                    if not(insideConstrainedOut):
                        d_8_chunkBudget_: int
                        d_8_chunkBudget_ = (maxSteps) - (d_2_steps_)
                        if (d_8_chunkBudget_) > (20):
                            d_8_chunkBudget_ = 20
                        if (d_8_chunkBudget_) == (0):
                            raise _dafny.Break("0")
                        d_9_chunkGenerated_: _dafny.Seq
                        d_10_stoppedOnOpen_: bool
                        d_11_stoppedOnEos_: bool
                        d_12_chunkSteps_: int
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: bool
                        out6_: int
                        out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_9_chunkGenerated_ = out3_
                        d_10_stoppedOnOpen_ = out4_
                        d_11_stoppedOnEos_ = out5_
                        d_12_chunkSteps_ = out6_
                        generated = d_9_chunkGenerated_
                        d_2_steps_ = (d_2_steps_) + (d_12_chunkSteps_)
                        if d_11_stoppedOnEos_:
                            raise _dafny.Break("0")
                        if d_10_stoppedOnOpen_:
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            generated = out7_
                            insideConstrainedOut = out8_
                            currentConstrainedOut = out9_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
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
                        d_2_steps_ = (d_2_steps_) + (1)
                    elif True:
                        d_16_constrainedPrompt_: _dafny.Seq
                        d_16_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_17_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        d_18_sinceLastClose_: int
                        out13_: int
                        out13_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                        d_18_sinceLastClose_ = out13_
                        if (d_18_sinceLastClose_) <= (3):
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), eosToken)
                            d_17_next_ = out14_
                        elif True:
                            out15_: _dafny.Seq
                            out15_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('15e-1'), eosToken)
                            d_17_next_ = out15_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_17_next_) == (eosToken):
                            d_19_closeBudget_: int
                            d_19_closeBudget_ = (maxSteps) - (d_2_steps_)
                            if (d_19_closeBudget_) >= (1):
                                d_20_cg_: _dafny.Seq
                                d_21_ci_: bool
                                d_22_cc_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_19_closeBudget_)
                                d_20_cg_ = out16_
                                d_21_ci_ = out17_
                                d_22_cc_ = out18_
                                generated = d_20_cg_
                                insideConstrainedOut = d_21_ci_
                                currentConstrainedOut = d_22_cc_
                                d_2_steps_ = maxSteps
                            raise _dafny.Break("0")
                        elif True:
                            d_23_appendedGenerated_: _dafny.Seq
                            d_24_appendedInside_: bool
                            d_25_appendedCurrent_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: _dafny.Seq
                            out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                            d_23_appendedGenerated_ = out19_
                            d_24_appendedInside_ = out20_
                            d_25_appendedCurrent_ = out21_
                            generated = d_23_appendedGenerated_
                            insideConstrainedOut = d_24_appendedInside_
                            currentConstrainedOut = d_25_appendedCurrent_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

