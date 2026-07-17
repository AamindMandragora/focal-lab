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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one SQL query. Write concise, correct SQL. Avoid repetition. Use only the provided schema tables and columns."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        if (maxSteps) == (0):
            cost = 0
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        d_2_steps_: int
        d_2_steps_ = 0
        if not(insideConstrainedOut):
            d_3_og_: _dafny.Seq
            d_4_oi_: bool
            d_5_oc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_3_og_ = out0_
            d_4_oi_ = out1_
            d_5_oc_ = out2_
            generated = d_3_og_
            insideConstrainedOut = d_4_oi_
            currentConstrainedOut = d_5_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        d_6_closeReserve_: int
        d_6_closeReserve_ = 3
        with _dafny.label("0"):
            while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    d_7_cg1_: _dafny.Seq
                    d_8_ci1_: bool
                    d_9_cc1_: _dafny.Seq
                    d_10_closed1_: bool
                    out3_: _dafny.Seq
                    out4_: bool
                    out5_: _dafny.Seq
                    out6_: bool
                    out3_, out4_, out5_, out6_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_7_cg1_ = out3_
                    d_8_ci1_ = out4_
                    d_9_cc1_ = out5_
                    d_10_closed1_ = out6_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if d_10_closed1_:
                        generated = d_7_cg1_
                        insideConstrainedOut = d_8_ci1_
                        currentConstrainedOut = d_9_cc1_
                        raise _dafny.Break("0")
                    if ((d_2_steps_) + (d_6_closeReserve_)) >= (maxSteps):
                        d_11_rg_: _dafny.Seq
                        d_12_rc_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: _dafny.Seq
                        out7_, out8_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_11_rg_ = out7_
                        d_12_rc_ = out8_
                        generated = d_11_rg_
                        currentConstrainedOut = d_12_rc_
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                            d_13_cg2_: _dafny.Seq
                            d_14_ci2_: bool
                            d_15_cc2_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_13_cg2_ = out9_
                            d_14_ci2_ = out10_
                            d_15_cc2_ = out11_
                            generated = d_13_cg2_
                            insideConstrainedOut = d_14_ci2_
                            currentConstrainedOut = d_15_cc2_
                            d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("0")
                    if (d_2_steps_) < (maxSteps):
                        d_16_constrainedPrompt_: _dafny.Seq
                        d_16_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_17_next_: _dafny.Seq
                        out12_: _dafny.Seq
                        out12_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                        d_17_next_ = out12_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_17_next_) == (eosToken):
                            d_18_rg2_: _dafny.Seq
                            d_19_rc2_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: _dafny.Seq
                            out13_, out14_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_18_rg2_ = out13_
                            d_19_rc2_ = out14_
                            generated = d_18_rg2_
                            currentConstrainedOut = d_19_rc2_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                d_20_cg3_: _dafny.Seq
                                d_21_ci3_: bool
                                d_22_cc3_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_20_cg3_ = out15_
                                d_21_ci3_ = out16_
                                d_22_cc3_ = out17_
                                generated = d_20_cg3_
                                insideConstrainedOut = d_21_ci3_
                                currentConstrainedOut = d_22_cc3_
                                d_2_steps_ = (d_2_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_23_isComplete_: bool
                            d_23_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            d_24_isValid_: bool
                            out18_: bool
                            out18_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_17_next_)
                            d_24_isValid_ = out18_
                            if (not(d_23_isComplete_)) and (d_24_isValid_):
                                d_25_ag_: _dafny.Seq
                                d_26_ai_: bool
                                d_27_ac_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                d_25_ag_ = out19_
                                d_26_ai_ = out20_
                                d_27_ac_ = out21_
                                generated = d_25_ag_
                                insideConstrainedOut = d_26_ai_
                                currentConstrainedOut = d_27_ac_
                            elif (d_23_isComplete_) and ((d_2_steps_) < (maxSteps)):
                                d_28_cg4_: _dafny.Seq
                                d_29_ci4_: bool
                                d_30_cc4_: _dafny.Seq
                                out22_: _dafny.Seq
                                out23_: bool
                                out24_: _dafny.Seq
                                out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_28_cg4_ = out22_
                                d_29_ci4_ = out23_
                                d_30_cc4_ = out24_
                                generated = d_28_cg4_
                                insideConstrainedOut = d_29_ci4_
                                currentConstrainedOut = d_30_cc4_
                                d_2_steps_ = (d_2_steps_) + (1)
                                raise _dafny.Break("0")
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_31_rg3_: _dafny.Seq
            d_32_rc3_: _dafny.Seq
            out25_: _dafny.Seq
            out26_: _dafny.Seq
            out25_, out26_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_31_rg3_ = out25_
            d_32_rc3_ = out26_
            generated = d_31_rg3_
            currentConstrainedOut = d_32_rc3_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                d_33_cg5_: _dafny.Seq
                d_34_ci5_: bool
                d_35_cc5_: _dafny.Seq
                out27_: _dafny.Seq
                out28_: bool
                out29_: _dafny.Seq
                out27_, out28_, out29_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_33_cg5_ = out27_
                d_34_ci5_ = out28_
                d_35_cc5_ = out29_
                generated = d_33_cg5_
                insideConstrainedOut = d_34_ci5_
                currentConstrainedOut = d_35_cc5_
                d_2_steps_ = (d_2_steps_) + (1)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

