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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly: SQL: <<single SQL SELECT query>>. Use only the tables and columns from the schema. Write simple, direct SQL. No explanation."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_3_chunkBudget_: int
            if (maxSteps) >= (15):
                d_3_chunkBudget_ = 15
            elif True:
                d_3_chunkBudget_ = maxSteps
            d_4_remaining_: int
            d_4_remaining_ = (maxSteps) - (d_2_steps_)
            d_5_actualBudget_: int
            if (d_3_chunkBudget_) <= (d_4_remaining_):
                d_5_actualBudget_ = d_3_chunkBudget_
            elif True:
                d_5_actualBudget_ = d_4_remaining_
            if (d_5_actualBudget_) >= (1):
                d_6_cg_: _dafny.Seq
                d_7_stoppedOnOpenSpan_: bool
                d_8_stoppedOnEos_: bool
                d_9_stepsUsed_: int
                out0_: _dafny.Seq
                out1_: bool
                out2_: bool
                out3_: int
                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_actualBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                d_6_cg_ = out0_
                d_7_stoppedOnOpenSpan_ = out1_
                d_8_stoppedOnEos_ = out2_
                d_9_stepsUsed_ = out3_
                d_2_steps_ = (d_2_steps_) + (d_9_stepsUsed_)
                generated = d_6_cg_
                if d_8_stoppedOnEos_:
                    cost = d_2_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                if d_7_stoppedOnOpenSpan_:
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
        d_10_closeReserve_: int
        if (maxSteps) >= (40):
            d_10_closeReserve_ = 20
        elif (maxSteps) >= (20):
            d_10_closeReserve_ = 10
        elif (maxSteps) >= (5):
            d_10_closeReserve_ = 3
        elif True:
            d_10_closeReserve_ = 1
        d_11_tokenIter_: int
        d_11_tokenIter_ = 0
        with _dafny.label("0"):
            while ((insideConstrainedOut) and ((d_2_steps_) < (maxSteps))) and (((maxSteps) - (d_2_steps_)) > (d_10_closeReserve_)):
                with _dafny.c_label("0"):
                    d_12_cg2_: _dafny.Seq
                    d_13_ci2_: bool
                    d_14_cc2_: _dafny.Seq
                    d_15_closed2_: bool
                    out10_: _dafny.Seq
                    out11_: bool
                    out12_: _dafny.Seq
                    out13_: bool
                    out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_12_cg2_ = out10_
                    d_13_ci2_ = out11_
                    d_14_cc2_ = out12_
                    d_15_closed2_ = out13_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if d_15_closed2_:
                        generated = d_12_cg2_
                        insideConstrainedOut = d_13_ci2_
                        currentConstrainedOut = d_14_cc2_
                    elif True:
                        d_16_stable_: _dafny.Seq
                        d_16_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_17_constrainedPrompt_: _dafny.Seq
                        d_17_constrainedPrompt_ = (prompt) + (d_16_stable_)
                        d_18_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if (_dafny.euclidian_modulus(d_11_tokenIter_, 3)) == (0):
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('15e-1'), eosToken)
                            d_18_next_ = out14_
                        elif True:
                            out15_: _dafny.Seq
                            out15_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 15, eosToken)
                            d_18_next_ = out15_
                        d_11_tokenIter_ = (d_11_tokenIter_) + (1)
                        if (d_18_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_19_ag_: _dafny.Seq
                            d_20_ai_: bool
                            d_21_ac_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                            d_19_ag_ = out16_
                            d_20_ai_ = out17_
                            d_21_ac_ = out18_
                            generated = d_19_ag_
                            insideConstrainedOut = d_20_ai_
                            currentConstrainedOut = d_21_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_22_closeBudget_: int
            d_22_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_23_cg3_: _dafny.Seq
            d_24_ci3_: bool
            d_25_cc3_: _dafny.Seq
            out19_: _dafny.Seq
            out20_: bool
            out21_: _dafny.Seq
            out19_, out20_, out21_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_22_closeBudget_)
            d_23_cg3_ = out19_
            d_24_ci3_ = out20_
            d_25_cc3_ = out21_
            generated = d_23_cg3_
            insideConstrainedOut = d_24_ci3_
            currentConstrainedOut = d_25_cc3_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

