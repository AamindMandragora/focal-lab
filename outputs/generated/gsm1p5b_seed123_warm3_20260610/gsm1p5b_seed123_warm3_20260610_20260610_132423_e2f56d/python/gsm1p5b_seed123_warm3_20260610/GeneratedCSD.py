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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. For each calculation and for the final answer, write a short arithmetic expression inside << >> delimiters, e.g. <<3+5=8>>. Keep each expression brief and complete.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        d_3_maxSpanTokens_: int
        d_3_maxSpanTokens_ = 35
        d_4_spanTokens_: int
        d_4_spanTokens_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_5_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_5_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                            if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_4_spanTokens_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_6_cg_: _dafny.Seq
                        d_7_ci_: bool
                        d_8_cc_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_6_cg_ = out1_
                        d_7_ci_ = out2_
                        d_8_cc_ = out3_
                        generated = d_6_cg_
                        insideConstrainedOut = d_7_ci_
                        currentConstrainedOut = d_8_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_4_spanTokens_ = 0
                    elif (d_4_spanTokens_) >= (d_3_maxSpanTokens_):
                        d_9_rg_: _dafny.Seq
                        d_10_rc_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: _dafny.Seq
                        out4_, out5_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_9_rg_ = out4_
                        d_10_rc_ = out5_
                        generated = d_9_rg_
                        currentConstrainedOut = d_10_rc_
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_11_cg2_: _dafny.Seq
                            d_12_ci2_: bool
                            d_13_cc2_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_11_cg2_ = out6_
                            d_12_ci2_ = out7_
                            d_13_cc2_ = out8_
                            generated = d_11_cg2_
                            insideConstrainedOut = d_12_ci2_
                            currentConstrainedOut = d_13_cc2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            if (d_1_steps_) < (maxSteps):
                                d_14_next2_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                d_14_next2_ = out9_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_14_next2_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_14_next2_]))
                                    if (d_14_next2_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                        insideConstrainedOut = True
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                        d_4_spanTokens_ = 0
                        d_4_spanTokens_ = 0
                    elif True:
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_16_validCount_: int
                        out10_: int
                        out10_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_16_validCount_ = out10_
                        d_17_next3_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if (d_16_validCount_) <= (d_2_narrowThreshold_):
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), eosToken)
                            d_17_next3_ = out11_
                        elif True:
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('4e0'), eosToken)
                            d_17_next3_ = out12_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_4_spanTokens_ = (d_4_spanTokens_) + (1)
                        if (d_17_next3_) == (eosToken):
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                if (d_1_steps_) < (maxSteps):
                                    d_18_cg3_: _dafny.Seq
                                    d_19_ci3_: bool
                                    d_20_cc3_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_18_cg3_ = out13_
                                    d_19_ci3_ = out14_
                                    d_20_cc3_ = out15_
                                    generated = d_18_cg3_
                                    insideConstrainedOut = d_19_ci3_
                                    currentConstrainedOut = d_20_cc3_
                                    d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_21_rg2_: _dafny.Seq
                                d_22_rc2_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: _dafny.Seq
                                out16_, out17_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_21_rg2_ = out16_
                                d_22_rc2_ = out17_
                                generated = d_21_rg2_
                                currentConstrainedOut = d_22_rc2_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_23_cg4_: _dafny.Seq
                                    d_24_ci4_: bool
                                    d_25_cc4_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_23_cg4_ = out18_
                                    d_24_ci4_ = out19_
                                    d_25_cc4_ = out20_
                                    generated = d_23_cg4_
                                    insideConstrainedOut = d_24_ci4_
                                    currentConstrainedOut = d_25_cc4_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            raise _dafny.Break("0")
                        elif True:
                            d_26_ag_: _dafny.Seq
                            d_27_ai_: bool
                            d_28_ac_: _dafny.Seq
                            out21_: _dafny.Seq
                            out22_: bool
                            out23_: _dafny.Seq
                            out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next3_)
                            d_26_ag_ = out21_
                            d_27_ai_ = out22_
                            d_28_ac_ = out23_
                            generated = d_26_ag_
                            insideConstrainedOut = d_27_ai_
                            currentConstrainedOut = d_28_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

