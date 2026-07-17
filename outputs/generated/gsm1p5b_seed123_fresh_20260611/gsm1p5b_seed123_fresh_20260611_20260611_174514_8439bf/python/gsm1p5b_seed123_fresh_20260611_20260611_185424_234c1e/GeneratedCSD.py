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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. Show your reasoning, then wrap only the final arithmetic expression in << >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanSteps_: int
        d_2_spanSteps_ = 0
        d_3_maxSpanSteps_: int
        d_3_maxSpanSteps_ = 10
        d_4_chunkSize_: int
        d_4_chunkSize_ = 25
        if ((maxSteps) > (0)) and (not(insideConstrainedOut)):
            d_5_firstNext_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_5_firstNext_ = out0_
            d_1_steps_ = (d_1_steps_) + (1)
            if (d_5_firstNext_) == (eosToken):
                cost = d_1_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            elif True:
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_firstNext_]))
                if (d_5_firstNext_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                    d_6_eg0_: _dafny.Seq
                    d_7_ei0_: bool
                    d_8_ec0_: _dafny.Seq
                    out1_: _dafny.Seq
                    out2_: bool
                    out3_: _dafny.Seq
                    out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                    d_6_eg0_ = out1_
                    d_7_ei0_ = out2_
                    d_8_ec0_ = out3_
                    generated = d_6_eg0_
                    insideConstrainedOut = d_7_ei0_
                    currentConstrainedOut = d_8_ec0_
                    d_2_spanSteps_ = 0
        elif ((maxSteps) > (0)) and (insideConstrainedOut):
            d_9_cg0_: _dafny.Seq
            d_10_ci0_: bool
            d_11_cc0_: _dafny.Seq
            d_12_closed0_: bool
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out7_: bool
            out4_, out5_, out6_, out7_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
            d_9_cg0_ = out4_
            d_10_ci0_ = out5_
            d_11_cc0_ = out6_
            d_12_closed0_ = out7_
            d_1_steps_ = (d_1_steps_) + (1)
            if d_12_closed0_:
                generated = d_9_cg0_
                insideConstrainedOut = d_10_ci0_
                currentConstrainedOut = d_11_cc0_
                d_2_spanSteps_ = 0
            elif True:
                d_13_constrainedPrompt0_: _dafny.Seq
                d_13_constrainedPrompt0_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_14_next0_: _dafny.Seq
                out8_: _dafny.Seq
                out8_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_13_constrainedPrompt0_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                d_14_next0_ = out8_
                if (d_14_next0_) == (eosToken):
                    d_15_rg0_: _dafny.Seq
                    d_16_rc0_: _dafny.Seq
                    out9_: _dafny.Seq
                    out10_: _dafny.Seq
                    out9_, out10_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                    d_15_rg0_ = out9_
                    d_16_rc0_ = out10_
                    generated = d_15_rg0_
                    currentConstrainedOut = d_16_rc0_
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                        d_17_closedG0_: _dafny.Seq
                        d_18_closedI0_: bool
                        d_19_closedC0_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: _dafny.Seq
                        out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_17_closedG0_ = out11_
                        d_18_closedI0_ = out12_
                        d_19_closedC0_ = out13_
                        generated = d_17_closedG0_
                        insideConstrainedOut = d_18_closedI0_
                        currentConstrainedOut = d_19_closedC0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanSteps_ = 0
                    elif True:
                        insideConstrainedOut = False
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_2_spanSteps_ = 0
                    cost = d_1_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                elif True:
                    d_20_ag0_: _dafny.Seq
                    d_21_ai0_: bool
                    d_22_ac0_: _dafny.Seq
                    out14_: _dafny.Seq
                    out15_: bool
                    out16_: _dafny.Seq
                    out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next0_)
                    d_20_ag0_ = out14_
                    d_21_ai0_ = out15_
                    d_22_ac0_ = out16_
                    generated = d_20_ag0_
                    insideConstrainedOut = d_21_ai0_
                    currentConstrainedOut = d_22_ac0_
                    d_2_spanSteps_ = (d_2_spanSteps_) + (1)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_23_remaining_: int
                        d_23_remaining_ = (maxSteps) - (d_1_steps_)
                        if (d_23_remaining_) < (3):
                            d_24_next1_: _dafny.Seq
                            out17_: _dafny.Seq
                            out17_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_24_next1_ = out17_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_24_next1_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_24_next1_]))
                                if (d_24_next1_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_25_eg1_: _dafny.Seq
                                    d_26_ei1_: bool
                                    d_27_ec1_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_25_eg1_ = out18_
                                    d_26_ei1_ = out19_
                                    d_27_ec1_ = out20_
                                    generated = d_25_eg1_
                                    insideConstrainedOut = d_26_ei1_
                                    currentConstrainedOut = d_27_ec1_
                                    d_2_spanSteps_ = 0
                        elif True:
                            d_28_budget_: int
                            if ((d_23_remaining_) - (2)) < (d_4_chunkSize_):
                                d_28_budget_ = (d_23_remaining_) - (2)
                            elif True:
                                d_28_budget_ = d_4_chunkSize_
                            d_29_chunkGenerated_: _dafny.Seq
                            d_30_stoppedOnOpenSpan_: bool
                            d_31_stoppedOnEos_: bool
                            d_32_stepsUsed_: int
                            out21_: _dafny.Seq
                            out22_: bool
                            out23_: bool
                            out24_: int
                            out21_, out22_, out23_, out24_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_28_budget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_29_chunkGenerated_ = out21_
                            d_30_stoppedOnOpenSpan_ = out22_
                            d_31_stoppedOnEos_ = out23_
                            d_32_stepsUsed_ = out24_
                            d_1_steps_ = (d_1_steps_) + (d_32_stepsUsed_)
                            generated = d_29_chunkGenerated_
                            if d_30_stoppedOnOpenSpan_:
                                d_33_eg2_: _dafny.Seq
                                d_34_ei2_: bool
                                d_35_ec2_: _dafny.Seq
                                out25_: _dafny.Seq
                                out26_: bool
                                out27_: _dafny.Seq
                                out25_, out26_, out27_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_33_eg2_ = out25_
                                d_34_ei2_ = out26_
                                d_35_ec2_ = out27_
                                generated = d_33_eg2_
                                insideConstrainedOut = d_34_ei2_
                                currentConstrainedOut = d_35_ec2_
                                d_2_spanSteps_ = 0
                            elif d_31_stoppedOnEos_:
                                raise _dafny.Break("0")
                    elif True:
                        d_36_remaining3_: int
                        d_36_remaining3_ = (maxSteps) - (d_1_steps_)
                        d_37_shouldForceClose_: bool
                        d_37_shouldForceClose_ = ((d_2_spanSteps_) >= (d_3_maxSpanSteps_)) or ((d_36_remaining3_) <= (1))
                        if d_37_shouldForceClose_:
                            d_38_rg3_: _dafny.Seq
                            d_39_rc3_: _dafny.Seq
                            out28_: _dafny.Seq
                            out29_: _dafny.Seq
                            out28_, out29_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_38_rg3_ = out28_
                            d_39_rc3_ = out29_
                            generated = d_38_rg3_
                            currentConstrainedOut = d_39_rc3_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_40_closedG3_: _dafny.Seq
                                d_41_closedI3_: bool
                                d_42_closedC3_: _dafny.Seq
                                out30_: _dafny.Seq
                                out31_: bool
                                out32_: _dafny.Seq
                                out30_, out31_, out32_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_40_closedG3_ = out30_
                                d_41_closedI3_ = out31_
                                d_42_closedC3_ = out32_
                                generated = d_40_closedG3_
                                insideConstrainedOut = d_41_closedI3_
                                currentConstrainedOut = d_42_closedC3_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_spanSteps_ = 0
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_spanSteps_ = 0
                                if (d_1_steps_) < (maxSteps):
                                    d_43_dummy_: _dafny.Seq
                                    out33_: _dafny.Seq
                                    out33_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                    d_43_dummy_ = out33_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_43_dummy_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_43_dummy_]))
                                        if (d_43_dummy_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                            d_44_eg4_: _dafny.Seq
                                            d_45_ei4_: bool
                                            d_46_ec4_: _dafny.Seq
                                            out34_: _dafny.Seq
                                            out35_: bool
                                            out36_: _dafny.Seq
                                            out34_, out35_, out36_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                            d_44_eg4_ = out34_
                                            d_45_ei4_ = out35_
                                            d_46_ec4_ = out36_
                                            generated = d_44_eg4_
                                            insideConstrainedOut = d_45_ei4_
                                            currentConstrainedOut = d_46_ec4_
                                            d_2_spanSteps_ = 0
                        elif True:
                            d_47_cg5_: _dafny.Seq
                            d_48_ci5_: bool
                            d_49_cc5_: _dafny.Seq
                            d_50_closed5_: bool
                            out37_: _dafny.Seq
                            out38_: bool
                            out39_: _dafny.Seq
                            out40_: bool
                            out37_, out38_, out39_, out40_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_47_cg5_ = out37_
                            d_48_ci5_ = out38_
                            d_49_cc5_ = out39_
                            d_50_closed5_ = out40_
                            if d_50_closed5_:
                                d_1_steps_ = (d_1_steps_) + (1)
                                generated = d_47_cg5_
                                insideConstrainedOut = d_48_ci5_
                                currentConstrainedOut = d_49_cc5_
                                d_2_spanSteps_ = 0
                            elif True:
                                d_51_constrainedPrompt5_: _dafny.Seq
                                d_51_constrainedPrompt5_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_52_next5_: _dafny.Seq
                                out41_: _dafny.Seq
                                out41_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_51_constrainedPrompt5_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_52_next5_ = out41_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                                if (d_52_next5_) == (eosToken):
                                    d_53_rg5_: _dafny.Seq
                                    d_54_rc5_: _dafny.Seq
                                    out42_: _dafny.Seq
                                    out43_: _dafny.Seq
                                    out42_, out43_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                    d_53_rg5_ = out42_
                                    d_54_rc5_ = out43_
                                    generated = d_53_rg5_
                                    currentConstrainedOut = d_54_rc5_
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_55_closedG5_: _dafny.Seq
                                        d_56_closedI5_: bool
                                        d_57_closedC5_: _dafny.Seq
                                        out44_: _dafny.Seq
                                        out45_: bool
                                        out46_: _dafny.Seq
                                        out44_, out45_, out46_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_55_closedG5_ = out44_
                                        d_56_closedI5_ = out45_
                                        d_57_closedC5_ = out46_
                                        generated = d_55_closedG5_
                                        insideConstrainedOut = d_56_closedI5_
                                        currentConstrainedOut = d_57_closedC5_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        d_2_spanSteps_ = 0
                                    elif True:
                                        insideConstrainedOut = False
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                        d_2_spanSteps_ = 0
                                    raise _dafny.Break("0")
                                elif True:
                                    d_58_ag5_: _dafny.Seq
                                    d_59_ai5_: bool
                                    d_60_ac5_: _dafny.Seq
                                    out47_: _dafny.Seq
                                    out48_: bool
                                    out49_: _dafny.Seq
                                    out47_, out48_, out49_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_52_next5_)
                                    d_58_ag5_ = out47_
                                    d_59_ai5_ = out48_
                                    d_60_ac5_ = out49_
                                    generated = d_58_ag5_
                                    insideConstrainedOut = d_59_ai5_
                                    currentConstrainedOut = d_60_ac5_
                    pass
            pass
        if insideConstrainedOut:
            d_61_rg6_: _dafny.Seq
            d_62_rc6_: _dafny.Seq
            out50_: _dafny.Seq
            out51_: _dafny.Seq
            out50_, out51_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_61_rg6_ = out50_
            d_62_rc6_ = out51_
            generated = d_61_rg6_
            currentConstrainedOut = d_62_rc6_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_63_closedG6_: _dafny.Seq
                d_64_closedI6_: bool
                d_65_closedC6_: _dafny.Seq
                out52_: _dafny.Seq
                out53_: bool
                out54_: _dafny.Seq
                out52_, out53_, out54_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_63_closedG6_ = out52_
                d_64_closedI6_ = out53_
                d_65_closedC6_ = out54_
                generated = d_63_closedG6_
                insideConstrainedOut = d_64_closedI6_
                currentConstrainedOut = d_65_closedC6_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

