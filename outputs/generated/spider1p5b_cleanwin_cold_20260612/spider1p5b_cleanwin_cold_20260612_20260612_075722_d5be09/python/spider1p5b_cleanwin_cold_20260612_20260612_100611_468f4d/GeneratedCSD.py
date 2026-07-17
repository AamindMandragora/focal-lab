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
        d_1_steps_: int
        d_1_steps_ = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write a concise SQL query. Use the simplest correct SQL that answers the question. Output: SQL: <<query>>")))
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_2_chunkMax_: int
            d_2_chunkMax_ = 3
            if (d_2_chunkMax_) > ((maxSteps) - (d_1_steps_)):
                d_2_chunkMax_ = (maxSteps) - (d_1_steps_)
            if (d_2_chunkMax_) > (0):
                d_3_genOut_: _dafny.Seq
                d_4_stoppedOnOpen_: bool
                d_5_stoppedOnEos_: bool
                d_6_stepsUsed_: int
                out0_: _dafny.Seq
                out1_: bool
                out2_: bool
                out3_: int
                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkMax_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                d_3_genOut_ = out0_
                d_4_stoppedOnOpen_ = out1_
                d_5_stoppedOnEos_ = out2_
                d_6_stepsUsed_ = out3_
                d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed_)
                generated = d_3_genOut_
                if d_5_stoppedOnEos_:
                    cost = d_1_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                if d_4_stoppedOnOpen_:
                    d_7_g2_: _dafny.Seq
                    d_8_i2_: bool
                    d_9_c2_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                    d_7_g2_ = out4_
                    d_8_i2_ = out5_
                    d_9_c2_ = out6_
                    generated = d_7_g2_
                    insideConstrainedOut = d_8_i2_
                    currentConstrainedOut = d_9_c2_
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_10_g2_: _dafny.Seq
            d_11_i2_: bool
            d_12_c2_: _dafny.Seq
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_10_g2_ = out7_
            d_11_i2_ = out8_
            d_12_c2_ = out9_
            generated = d_10_g2_
            insideConstrainedOut = d_11_i2_
            currentConstrainedOut = d_12_c2_
            d_1_steps_ = (d_1_steps_) + (1)
        d_13_hardCap_: int
        d_13_hardCap_ = 30
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    d_14_repDetected_: bool
                    d_14_repDetected_ = False
                    if (len(currentConstrainedOut)) >= (8):
                        d_15_lastTok_: _dafny.Seq
                        d_15_lastTok_ = (currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]
                        d_16_occCount_: int
                        out10_: int
                        out10_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, d_15_lastTok_)
                        d_16_occCount_ = out10_
                        if (d_16_occCount_) >= (4):
                            d_14_repDetected_ = True
                    if (((len(currentConstrainedOut)) >= (d_13_hardCap_)) or (d_14_repDetected_)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                        d_17_rg_: _dafny.Seq
                        d_18_rc_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: _dafny.Seq
                        out11_, out12_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_17_rg_ = out11_
                        d_18_rc_ = out12_
                        generated = d_17_rg_
                        currentConstrainedOut = d_18_rc_
                        insideConstrainedOut = True
                        if ((parser).IsCompletePrefix(d_18_rc_)) and ((d_1_steps_) < (maxSteps)):
                            d_19_cg3_: _dafny.Seq
                            d_20_ci3_: bool
                            d_21_cc3_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_19_cg3_ = out13_
                            d_20_ci3_ = out14_
                            d_21_cc3_ = out15_
                            generated = d_19_cg3_
                            insideConstrainedOut = d_20_ci3_
                            currentConstrainedOut = d_21_cc3_
                            d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    d_22_cg_: _dafny.Seq
                    d_23_ci_: bool
                    d_24_cc_: _dafny.Seq
                    d_25_closed_: bool
                    out16_: _dafny.Seq
                    out17_: bool
                    out18_: _dafny.Seq
                    out19_: bool
                    out16_, out17_, out18_, out19_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_22_cg_ = out16_
                    d_23_ci_ = out17_
                    d_24_cc_ = out18_
                    d_25_closed_ = out19_
                    if d_25_closed_:
                        generated = d_22_cg_
                        insideConstrainedOut = d_23_ci_
                        currentConstrainedOut = d_24_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    if (d_1_steps_) >= (maxSteps):
                        raise _dafny.Break("0")
                    d_26_constrainedPrompt_: _dafny.Seq
                    d_26_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_27_next_: _dafny.Seq
                    out20_: _dafny.Seq
                    out20_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_26_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('4e0'), eosToken)
                    d_27_next_ = out20_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_27_next_) == (eosToken):
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_28_cg2_: _dafny.Seq
                            d_29_ci2_: bool
                            d_30_cc2_: _dafny.Seq
                            out21_: _dafny.Seq
                            out22_: bool
                            out23_: _dafny.Seq
                            out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_28_cg2_ = out21_
                            d_29_ci2_ = out22_
                            d_30_cc2_ = out23_
                            generated = d_28_cg2_
                            insideConstrainedOut = d_29_ci2_
                            currentConstrainedOut = d_30_cc2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_1_steps_) < (maxSteps):
                            d_31_rg_: _dafny.Seq
                            d_32_rc_: _dafny.Seq
                            out24_: _dafny.Seq
                            out25_: _dafny.Seq
                            out24_, out25_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_31_rg_ = out24_
                            d_32_rc_ = out25_
                            generated = d_31_rg_
                            currentConstrainedOut = d_32_rc_
                            insideConstrainedOut = True
                            if ((parser).IsCompletePrefix(d_32_rc_)) and ((d_1_steps_) < (maxSteps)):
                                d_33_cg4_: _dafny.Seq
                                d_34_ci4_: bool
                                d_35_cc4_: _dafny.Seq
                                out26_: _dafny.Seq
                                out27_: bool
                                out28_: _dafny.Seq
                                out26_, out27_, out28_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_33_cg4_ = out26_
                                d_34_ci4_ = out27_
                                d_35_cc4_ = out28_
                                generated = d_33_cg4_
                                insideConstrainedOut = d_34_ci4_
                                currentConstrainedOut = d_35_cc4_
                                d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_36_ag_: _dafny.Seq
                        d_37_ai_: bool
                        d_38_ac_: _dafny.Seq
                        out29_: _dafny.Seq
                        out30_: bool
                        out31_: _dafny.Seq
                        out29_, out30_, out31_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_next_)
                        d_36_ag_ = out29_
                        d_37_ai_ = out30_
                        d_38_ac_ = out31_
                        generated = d_36_ag_
                        insideConstrainedOut = d_37_ai_
                        currentConstrainedOut = d_38_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_39_rg_: _dafny.Seq
            d_40_rc_: _dafny.Seq
            out32_: _dafny.Seq
            out33_: _dafny.Seq
            out32_, out33_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_39_rg_ = out32_
            d_40_rc_ = out33_
            generated = d_39_rg_
            currentConstrainedOut = d_40_rc_
            insideConstrainedOut = True
            if ((parser).IsCompletePrefix(d_40_rc_)) and ((d_1_steps_) < (maxSteps)):
                d_41_cg3_: _dafny.Seq
                d_42_ci3_: bool
                d_43_cc3_: _dafny.Seq
                out34_: _dafny.Seq
                out35_: bool
                out36_: _dafny.Seq
                out34_, out35_, out36_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_41_cg3_ = out34_
                d_42_ci3_ = out35_
                d_43_cc3_ = out36_
                generated = d_41_cg3_
                insideConstrainedOut = d_42_ci3_
                currentConstrainedOut = d_43_cc3_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

