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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly: SQL: <<SELECT ...>> using ONLY the exact table names and column names from the schema provided. Complete the full SQL query. Use correct JOINs, WHERE, GROUP BY as needed. Stop after >>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_3_chunkBudget_: int
            if (8) <= ((maxSteps) - (d_2_steps_)):
                d_3_chunkBudget_ = 8
            elif True:
                d_3_chunkBudget_ = (maxSteps) - (d_2_steps_)
            if (d_3_chunkBudget_) >= (1):
                d_4_cg_: _dafny.Seq
                d_5_stoppedOnOpenSpan_: bool
                d_6_stoppedOnEos_: bool
                d_7_stepsUsed_: int
                out0_: _dafny.Seq
                out1_: bool
                out2_: bool
                out3_: int
                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                d_4_cg_ = out0_
                d_5_stoppedOnOpenSpan_ = out1_
                d_6_stoppedOnEos_ = out2_
                d_7_stepsUsed_ = out3_
                d_2_steps_ = (d_2_steps_) + (d_7_stepsUsed_)
                generated = d_4_cg_
                if d_6_stoppedOnEos_:
                    cost = d_2_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                if d_5_stoppedOnOpenSpan_:
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                    generated = out4_
                    insideConstrainedOut = out5_
                    currentConstrainedOut = out6_
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            generated = out7_
            insideConstrainedOut = out8_
            currentConstrainedOut = out9_
            d_2_steps_ = (d_2_steps_) + (1)
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_8_rem_: int
            d_8_rem_ = (maxSteps) - (d_2_steps_)
            d_9_closeReserve_: int
            if (d_8_rem_) >= (50):
                d_9_closeReserve_ = 40
            elif (d_8_rem_) >= (20):
                d_9_closeReserve_ = 15
            elif (d_8_rem_) >= (5):
                d_9_closeReserve_ = 4
            elif True:
                d_9_closeReserve_ = d_8_rem_
            d_10_fillLimit_: int
            if (d_8_rem_) > (d_9_closeReserve_):
                d_10_fillLimit_ = (d_8_rem_) - (d_9_closeReserve_)
            elif True:
                d_10_fillLimit_ = 0
            d_11_schemaBoostTokens_: _dafny.Seq
            d_11_schemaBoostTokens_ = _dafny.SeqWithoutIsStrInference([])
            if (len(validTokenGroups)) >= (1):
                d_11_schemaBoostTokens_ = (validTokenGroups)[0]
            d_12_fillSteps_: int
            d_12_fillSteps_ = 0
            with _dafny.label("2_0"):
                while ((insideConstrainedOut) and ((d_12_fillSteps_) < (d_10_fillLimit_))) and ((d_2_steps_) < (maxSteps)):
                    with _dafny.c_label("2_0"):
                        d_13_cg0_: _dafny.Seq
                        d_14_ci0_: bool
                        d_15_cc0_: _dafny.Seq
                        d_16_closed0_: bool
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out13_: bool
                        out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_13_cg0_ = out10_
                        d_14_ci0_ = out11_
                        d_15_cc0_ = out12_
                        d_16_closed0_ = out13_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_12_fillSteps_ = (d_12_fillSteps_) + (1)
                        if d_16_closed0_:
                            generated = d_13_cg0_
                            insideConstrainedOut = d_14_ci0_
                            currentConstrainedOut = d_15_cc0_
                            raise _dafny.Break("2_0")
                        if ((insideConstrainedOut) and ((d_2_steps_) < (maxSteps))) and ((d_12_fillSteps_) < (d_10_fillLimit_)):
                            d_17_stable_: _dafny.Seq
                            d_17_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_18_constrainedPrompt_: _dafny.Seq
                            d_18_constrainedPrompt_ = (prompt) + (d_17_stable_)
                            (d_0_helpers_).BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'))
                            d_19_next_: _dafny.Seq
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, d_11_schemaBoostTokens_, _dafny.BigRational('3e0'), eosToken)
                            d_19_next_ = out14_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_12_fillSteps_ = (d_12_fillSteps_) + (1)
                            if (d_19_next_) == (eosToken):
                                raise _dafny.Break("2_0")
                            d_20_valid_: bool
                            out15_: bool
                            out15_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_19_next_)
                            d_20_valid_ = out15_
                            if d_20_valid_:
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                generated = out16_
                                insideConstrainedOut = out17_
                                currentConstrainedOut = out18_
                        pass
                pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_21_closeBudget_: int
            d_21_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_22_cg_: _dafny.Seq
            d_23_ci_: bool
            d_24_cc_: _dafny.Seq
            out19_: _dafny.Seq
            out20_: bool
            out21_: _dafny.Seq
            out19_, out20_, out21_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_21_closeBudget_)
            d_22_cg_ = out19_
            d_23_ci_ = out20_
            d_24_cc_ = out21_
            generated = d_22_cg_
            insideConstrainedOut = d_23_ci_
            currentConstrainedOut = d_24_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

