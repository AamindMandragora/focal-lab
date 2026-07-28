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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Use only plain variable names (letters and underscores, no curly braces) inside << >>. Write the final answer as a single arithmetic expression like <<n * m - k>>. Do not repeat yourself.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanSteps_: int
        d_2_spanSteps_ = 0
        d_3_maxSpanSteps_: int
        d_3_maxSpanSteps_ = 20
        d_4_chunkSize_: int
        d_4_chunkSize_ = 10
        d_5_spansOpened_: int
        d_5_spansOpened_ = 0
        d_6_maxSpans_: int
        d_6_maxSpans_ = 6
        if ((maxSteps) > (0)) and (not(insideConstrainedOut)):
            d_7_firstNext_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_7_firstNext_ = out0_
            d_1_steps_ = (d_1_steps_) + (1)
            if (d_7_firstNext_) == (eosToken):
                cost = d_1_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            elif True:
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_firstNext_]))
                if (d_7_firstNext_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                    d_5_spansOpened_ = (d_5_spansOpened_) + (1)
                    d_8_eg0_: _dafny.Seq
                    d_9_ei0_: bool
                    d_10_ec0_: _dafny.Seq
                    out1_: _dafny.Seq
                    out2_: bool
                    out3_: _dafny.Seq
                    out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                    d_8_eg0_ = out1_
                    d_9_ei0_ = out2_
                    d_10_ec0_ = out3_
                    generated = d_8_eg0_
                    insideConstrainedOut = d_9_ei0_
                    currentConstrainedOut = d_10_ec0_
                    d_2_spanSteps_ = 0
        elif ((maxSteps) > (0)) and (insideConstrainedOut):
            d_11_cg0_: _dafny.Seq
            d_12_ci0_: bool
            d_13_cc0_: _dafny.Seq
            d_14_closed0_: bool
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out7_: bool
            out4_, out5_, out6_, out7_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
            d_11_cg0_ = out4_
            d_12_ci0_ = out5_
            d_13_cc0_ = out6_
            d_14_closed0_ = out7_
            d_1_steps_ = (d_1_steps_) + (1)
            if d_14_closed0_:
                generated = d_11_cg0_
                insideConstrainedOut = d_12_ci0_
                currentConstrainedOut = d_13_cc0_
                d_2_spanSteps_ = 0
            elif True:
                d_15_constrainedPrompt0_: _dafny.Seq
                d_15_constrainedPrompt0_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_16_next0_: _dafny.Seq
                out8_: _dafny.Seq
                out8_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_15_constrainedPrompt0_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                d_16_next0_ = out8_
                if (d_16_next0_) == (eosToken):
                    d_17_rg0_: _dafny.Seq
                    d_18_rc0_: _dafny.Seq
                    out9_: _dafny.Seq
                    out10_: _dafny.Seq
                    out9_, out10_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                    d_17_rg0_ = out9_
                    d_18_rc0_ = out10_
                    generated = d_17_rg0_
                    currentConstrainedOut = d_18_rc0_
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                        d_19_closedG0_: _dafny.Seq
                        d_20_closedI0_: bool
                        d_21_closedC0_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: _dafny.Seq
                        out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_19_closedG0_ = out11_
                        d_20_closedI0_ = out12_
                        d_21_closedC0_ = out13_
                        generated = d_19_closedG0_
                        insideConstrainedOut = d_20_closedI0_
                        currentConstrainedOut = d_21_closedC0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanSteps_ = 0
                    elif True:
                        insideConstrainedOut = False
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_2_spanSteps_ = 0
                    cost = d_1_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                elif True:
                    d_22_ag0_: _dafny.Seq
                    d_23_ai0_: bool
                    d_24_ac0_: _dafny.Seq
                    out14_: _dafny.Seq
                    out15_: bool
                    out16_: _dafny.Seq
                    out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next0_)
                    d_22_ag0_ = out14_
                    d_23_ai0_ = out15_
                    d_24_ac0_ = out16_
                    generated = d_22_ag0_
                    insideConstrainedOut = d_23_ai0_
                    currentConstrainedOut = d_24_ac0_
                    d_2_spanSteps_ = (d_2_spanSteps_) + (1)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if (d_5_spansOpened_) >= (d_6_maxSpans_):
                        raise _dafny.Break("0")
                    if not(insideConstrainedOut):
                        d_25_remaining_: int
                        d_25_remaining_ = (maxSteps) - (d_1_steps_)
                        if (d_25_remaining_) < (3):
                            d_26_next1_: _dafny.Seq
                            out17_: _dafny.Seq
                            out17_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_26_next1_ = out17_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_26_next1_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_26_next1_]))
                                if (d_26_next1_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_5_spansOpened_ = (d_5_spansOpened_) + (1)
                                    d_27_eg1_: _dafny.Seq
                                    d_28_ei1_: bool
                                    d_29_ec1_: _dafny.Seq
                                    out18_: _dafny.Seq
                                    out19_: bool
                                    out20_: _dafny.Seq
                                    out18_, out19_, out20_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_27_eg1_ = out18_
                                    d_28_ei1_ = out19_
                                    d_29_ec1_ = out20_
                                    generated = d_27_eg1_
                                    insideConstrainedOut = d_28_ei1_
                                    currentConstrainedOut = d_29_ec1_
                                    d_2_spanSteps_ = 0
                        elif True:
                            d_30_budget_: int
                            if ((d_25_remaining_) - (2)) < (d_4_chunkSize_):
                                d_30_budget_ = (d_25_remaining_) - (2)
                            elif True:
                                d_30_budget_ = d_4_chunkSize_
                            d_31_chunkGenerated_: _dafny.Seq
                            d_32_stoppedOnOpenSpan_: bool
                            d_33_stoppedOnEos_: bool
                            d_34_stepsUsed_: int
                            out21_: _dafny.Seq
                            out22_: bool
                            out23_: bool
                            out24_: int
                            out21_, out22_, out23_, out24_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_30_budget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_31_chunkGenerated_ = out21_
                            d_32_stoppedOnOpenSpan_ = out22_
                            d_33_stoppedOnEos_ = out23_
                            d_34_stepsUsed_ = out24_
                            d_1_steps_ = (d_1_steps_) + (d_34_stepsUsed_)
                            generated = d_31_chunkGenerated_
                            if d_32_stoppedOnOpenSpan_:
                                d_5_spansOpened_ = (d_5_spansOpened_) + (1)
                                d_35_eg2_: _dafny.Seq
                                d_36_ei2_: bool
                                d_37_ec2_: _dafny.Seq
                                out25_: _dafny.Seq
                                out26_: bool
                                out27_: _dafny.Seq
                                out25_, out26_, out27_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_35_eg2_ = out25_
                                d_36_ei2_ = out26_
                                d_37_ec2_ = out27_
                                generated = d_35_eg2_
                                insideConstrainedOut = d_36_ei2_
                                currentConstrainedOut = d_37_ec2_
                                d_2_spanSteps_ = 0
                            elif d_33_stoppedOnEos_:
                                raise _dafny.Break("0")
                    elif True:
                        d_38_remaining3_: int
                        d_38_remaining3_ = (maxSteps) - (d_1_steps_)
                        d_39_shouldForceClose_: bool
                        d_39_shouldForceClose_ = ((d_2_spanSteps_) >= (d_3_maxSpanSteps_)) or ((d_38_remaining3_) <= (1))
                        if d_39_shouldForceClose_:
                            d_40_rg3_: _dafny.Seq
                            d_41_rc3_: _dafny.Seq
                            out28_: _dafny.Seq
                            out29_: _dafny.Seq
                            out28_, out29_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_40_rg3_ = out28_
                            d_41_rc3_ = out29_
                            generated = d_40_rg3_
                            currentConstrainedOut = d_41_rc3_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_42_closedG3_: _dafny.Seq
                                d_43_closedI3_: bool
                                d_44_closedC3_: _dafny.Seq
                                out30_: _dafny.Seq
                                out31_: bool
                                out32_: _dafny.Seq
                                out30_, out31_, out32_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_42_closedG3_ = out30_
                                d_43_closedI3_ = out31_
                                d_44_closedC3_ = out32_
                                generated = d_42_closedG3_
                                insideConstrainedOut = d_43_closedI3_
                                currentConstrainedOut = d_44_closedC3_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_spanSteps_ = 0
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_spanSteps_ = 0
                                if (d_1_steps_) < (maxSteps):
                                    d_45_dummy_: _dafny.Seq
                                    out33_: _dafny.Seq
                                    out33_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                    d_45_dummy_ = out33_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_45_dummy_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_45_dummy_]))
                                        if (d_45_dummy_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                            d_5_spansOpened_ = (d_5_spansOpened_) + (1)
                                            d_46_eg4_: _dafny.Seq
                                            d_47_ei4_: bool
                                            d_48_ec4_: _dafny.Seq
                                            out34_: _dafny.Seq
                                            out35_: bool
                                            out36_: _dafny.Seq
                                            out34_, out35_, out36_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                            d_46_eg4_ = out34_
                                            d_47_ei4_ = out35_
                                            d_48_ec4_ = out36_
                                            generated = d_46_eg4_
                                            insideConstrainedOut = d_47_ei4_
                                            currentConstrainedOut = d_48_ec4_
                                            d_2_spanSteps_ = 0
                        elif True:
                            d_49_cg5_: _dafny.Seq
                            d_50_ci5_: bool
                            d_51_cc5_: _dafny.Seq
                            d_52_closed5_: bool
                            out37_: _dafny.Seq
                            out38_: bool
                            out39_: _dafny.Seq
                            out40_: bool
                            out37_, out38_, out39_, out40_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_49_cg5_ = out37_
                            d_50_ci5_ = out38_
                            d_51_cc5_ = out39_
                            d_52_closed5_ = out40_
                            if d_52_closed5_:
                                d_1_steps_ = (d_1_steps_) + (1)
                                generated = d_49_cg5_
                                insideConstrainedOut = d_50_ci5_
                                currentConstrainedOut = d_51_cc5_
                                d_2_spanSteps_ = 0
                            elif True:
                                d_53_constrainedPrompt5_: _dafny.Seq
                                d_53_constrainedPrompt5_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_54_next5_: _dafny.Seq
                                out41_: _dafny.Seq
                                out41_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_53_constrainedPrompt5_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                d_54_next5_ = out41_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                                if (d_54_next5_) == (eosToken):
                                    d_55_rg5_: _dafny.Seq
                                    d_56_rc5_: _dafny.Seq
                                    out42_: _dafny.Seq
                                    out43_: _dafny.Seq
                                    out42_, out43_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                    d_55_rg5_ = out42_
                                    d_56_rc5_ = out43_
                                    generated = d_55_rg5_
                                    currentConstrainedOut = d_56_rc5_
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_57_closedG5_: _dafny.Seq
                                        d_58_closedI5_: bool
                                        d_59_closedC5_: _dafny.Seq
                                        out44_: _dafny.Seq
                                        out45_: bool
                                        out46_: _dafny.Seq
                                        out44_, out45_, out46_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_57_closedG5_ = out44_
                                        d_58_closedI5_ = out45_
                                        d_59_closedC5_ = out46_
                                        generated = d_57_closedG5_
                                        insideConstrainedOut = d_58_closedI5_
                                        currentConstrainedOut = d_59_closedC5_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        d_2_spanSteps_ = 0
                                    elif True:
                                        insideConstrainedOut = False
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                        d_2_spanSteps_ = 0
                                    raise _dafny.Break("0")
                                elif True:
                                    d_60_ag5_: _dafny.Seq
                                    d_61_ai5_: bool
                                    d_62_ac5_: _dafny.Seq
                                    out47_: _dafny.Seq
                                    out48_: bool
                                    out49_: _dafny.Seq
                                    out47_, out48_, out49_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_54_next5_)
                                    d_60_ag5_ = out47_
                                    d_61_ai5_ = out48_
                                    d_62_ac5_ = out49_
                                    generated = d_60_ag5_
                                    insideConstrainedOut = d_61_ai5_
                                    currentConstrainedOut = d_62_ac5_
                    pass
            pass
        if insideConstrainedOut:
            d_63_rg6_: _dafny.Seq
            d_64_rc6_: _dafny.Seq
            out50_: _dafny.Seq
            out51_: _dafny.Seq
            out50_, out51_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_63_rg6_ = out50_
            d_64_rc6_ = out51_
            generated = d_63_rg6_
            currentConstrainedOut = d_64_rc6_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_65_closedG6_: _dafny.Seq
                d_66_closedI6_: bool
                d_67_closedC6_: _dafny.Seq
                out52_: _dafny.Seq
                out53_: bool
                out54_: _dafny.Seq
                out52_, out53_, out54_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_65_closedG6_ = out52_
                d_66_closedI6_ = out53_
                d_67_closedC6_ = out54_
                generated = d_65_closedG6_
                insideConstrainedOut = d_66_closedI6_
                currentConstrainedOut = d_67_closedC6_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

