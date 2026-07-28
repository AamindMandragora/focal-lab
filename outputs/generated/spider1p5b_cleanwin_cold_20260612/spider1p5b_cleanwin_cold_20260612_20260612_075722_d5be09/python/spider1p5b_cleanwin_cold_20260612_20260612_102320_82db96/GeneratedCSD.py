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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single valid SQL query answering the question. Output exactly: SQL: <<SELECT ...>> and nothing else. Use simple SQL without subqueries when possible.")))
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_2_chunkMax_: int
            d_2_chunkMax_ = 5
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
        d_13_hardCap_ = 60
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                        d_14_cg_: _dafny.Seq
                        d_15_ci_: bool
                        d_16_cc_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_14_cg_ = out10_
                        d_15_ci_ = out11_
                        d_16_cc_ = out12_
                        generated = d_14_cg_
                        insideConstrainedOut = d_15_ci_
                        currentConstrainedOut = d_16_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    if ((len(currentConstrainedOut)) >= (d_13_hardCap_)) and (((d_1_steps_) + (2)) <= (maxSteps)):
                        d_17_rg_: _dafny.Seq
                        d_18_rc_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: _dafny.Seq
                        out13_, out14_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_17_rg_ = out13_
                        d_18_rc_ = out14_
                        generated = d_17_rg_
                        currentConstrainedOut = d_18_rc_
                        insideConstrainedOut = True
                        if ((parser).IsCompletePrefix(d_18_rc_)) and ((d_1_steps_) < (maxSteps)):
                            d_19_cg3_: _dafny.Seq
                            d_20_ci3_: bool
                            d_21_cc3_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_19_cg3_ = out15_
                            d_20_ci3_ = out16_
                            d_21_cc3_ = out17_
                            generated = d_19_cg3_
                            insideConstrainedOut = d_20_ci3_
                            currentConstrainedOut = d_21_cc3_
                            d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    if (d_1_steps_) >= (maxSteps):
                        raise _dafny.Break("0")
                    d_22_constrainedPrompt_: _dafny.Seq
                    d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_23_next_: _dafny.Seq
                    out18_: _dafny.Seq
                    out18_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, eosToken)
                    d_23_next_ = out18_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_23_next_) == (eosToken):
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_24_cg2_: _dafny.Seq
                            d_25_ci2_: bool
                            d_26_cc2_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: _dafny.Seq
                            out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_24_cg2_ = out19_
                            d_25_ci2_ = out20_
                            d_26_cc2_ = out21_
                            generated = d_24_cg2_
                            insideConstrainedOut = d_25_ci2_
                            currentConstrainedOut = d_26_cc2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_1_steps_) < (maxSteps):
                            d_27_rg_: _dafny.Seq
                            d_28_rc_: _dafny.Seq
                            out22_: _dafny.Seq
                            out23_: _dafny.Seq
                            out22_, out23_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_27_rg_ = out22_
                            d_28_rc_ = out23_
                            generated = d_27_rg_
                            currentConstrainedOut = d_28_rc_
                            insideConstrainedOut = True
                            if ((parser).IsCompletePrefix(d_28_rc_)) and ((d_1_steps_) < (maxSteps)):
                                d_29_cg4_: _dafny.Seq
                                d_30_ci4_: bool
                                d_31_cc4_: _dafny.Seq
                                out24_: _dafny.Seq
                                out25_: bool
                                out26_: _dafny.Seq
                                out24_, out25_, out26_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_29_cg4_ = out24_
                                d_30_ci4_ = out25_
                                d_31_cc4_ = out26_
                                generated = d_29_cg4_
                                insideConstrainedOut = d_30_ci4_
                                currentConstrainedOut = d_31_cc4_
                                d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_32_ag_: _dafny.Seq
                        d_33_ai_: bool
                        d_34_ac_: _dafny.Seq
                        out27_: _dafny.Seq
                        out28_: bool
                        out29_: _dafny.Seq
                        out27_, out28_, out29_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                        d_32_ag_ = out27_
                        d_33_ai_ = out28_
                        d_34_ac_ = out29_
                        generated = d_32_ag_
                        insideConstrainedOut = d_33_ai_
                        currentConstrainedOut = d_34_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_35_rg_: _dafny.Seq
            d_36_rc_: _dafny.Seq
            out30_: _dafny.Seq
            out31_: _dafny.Seq
            out30_, out31_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_35_rg_ = out30_
            d_36_rc_ = out31_
            if (parser).IsCompletePrefix(d_36_rc_):
                generated = d_35_rg_
                currentConstrainedOut = d_36_rc_
                insideConstrainedOut = True
                if (d_1_steps_) < (maxSteps):
                    d_37_cg3_: _dafny.Seq
                    d_38_ci3_: bool
                    d_39_cc3_: _dafny.Seq
                    out32_: _dafny.Seq
                    out33_: bool
                    out34_: _dafny.Seq
                    out32_, out33_, out34_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_37_cg3_ = out32_
                    d_38_ci3_ = out33_
                    d_39_cc3_ = out34_
                    generated = d_37_cg3_
                    insideConstrainedOut = d_38_ci3_
                    currentConstrainedOut = d_39_cc3_
                    d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

