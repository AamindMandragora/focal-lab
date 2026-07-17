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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. At the very end, write the final arithmetic expression inside << >> delimiters. Use ONLY numbers, variable names (letters/digits/underscore), +, -, *, /, (, ) inside the delimiters. Do NOT use ^, {, }, %, or other symbols.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_constrainedReserve_: int
        if (maxSteps) >= (50):
            d_2_constrainedReserve_ = 40
        elif (maxSteps) >= (10):
            d_2_constrainedReserve_ = _dafny.euclidian_division(maxSteps, 2)
        elif True:
            d_2_constrainedReserve_ = maxSteps
        d_3_freePhaseLimit_: int
        if (maxSteps) > (d_2_constrainedReserve_):
            d_3_freePhaseLimit_ = (maxSteps) - (d_2_constrainedReserve_)
        elif True:
            d_3_freePhaseLimit_ = 0
        d_4_minFreeTokens_: int
        if (d_3_freePhaseLimit_) > (60):
            d_4_minFreeTokens_ = 60
        elif True:
            d_4_minFreeTokens_ = d_3_freePhaseLimit_
        d_5_penaltyTokens_: _dafny.Seq
        d_5_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "^")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "{")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "}")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "%")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "**")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "//")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "^{")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "}^")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "^d")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "^n"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_1_steps_) >= (d_3_freePhaseLimit_):
                            d_6_og_: _dafny.Seq
                            d_7_oi_: bool
                            d_8_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_og_ = out0_
                            d_7_oi_ = out1_
                            d_8_oc_ = out2_
                            generated = d_6_og_
                            insideConstrainedOut = d_7_oi_
                            currentConstrainedOut = d_8_oc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_9_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                if ((d_1_steps_) >= (d_4_minFreeTokens_)) and (((d_1_steps_) + (2)) <= (maxSteps)):
                                    d_10_og_: _dafny.Seq
                                    d_11_oi_: bool
                                    d_12_oc_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_10_og_ = out4_
                                    d_11_oi_ = out5_
                                    d_12_oc_ = out6_
                                    generated = d_10_og_
                                    insideConstrainedOut = d_11_oi_
                                    currentConstrainedOut = d_12_oc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif (d_1_steps_) < (d_4_minFreeTokens_):
                                    raise _dafny.Break("0")
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_13_cg_: _dafny.Seq
                        d_14_ci_: bool
                        d_15_cc_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_13_cg_ = out7_
                        d_14_ci_ = out8_
                        d_15_cc_ = out9_
                        generated = d_13_cg_
                        insideConstrainedOut = d_14_ci_
                        currentConstrainedOut = d_15_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_16_narrow_: bool
                        out10_: bool
                        out10_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_16_narrow_ = out10_
                        if d_16_narrow_:
                            d_17_rg_: _dafny.Seq
                            d_18_rc_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: _dafny.Seq
                            out11_, out12_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_17_rg_ = out11_
                            d_18_rc_ = out12_
                            generated = d_17_rg_
                            currentConstrainedOut = d_18_rc_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                d_19_cg2_: _dafny.Seq
                                d_20_ci2_: bool
                                d_21_cc2_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_19_cg2_ = out13_
                                d_20_ci2_ = out14_
                                d_21_cc2_ = out15_
                                generated = d_19_cg2_
                                insideConstrainedOut = d_20_ci2_
                                currentConstrainedOut = d_21_cc2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_22_constrainedPrompt_: _dafny.Seq
                            d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_23_next_: _dafny.Seq
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_5_penaltyTokens_, _dafny.BigRational('3e0'), 12, eosToken)
                            d_23_next_ = out16_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_23_next_) == (eosToken):
                                d_24_rg_: _dafny.Seq
                                d_25_rc_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: _dafny.Seq
                                out17_, out18_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_24_rg_ = out17_
                                d_25_rc_ = out18_
                                generated = d_24_rg_
                                currentConstrainedOut = d_25_rc_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                    d_26_cg2_: _dafny.Seq
                                    d_27_ci2_: bool
                                    d_28_cc2_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out20_: bool
                                    out21_: _dafny.Seq
                                    out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_26_cg2_ = out19_
                                    d_27_ci2_ = out20_
                                    d_28_cc2_ = out21_
                                    generated = d_26_cg2_
                                    insideConstrainedOut = d_27_ci2_
                                    currentConstrainedOut = d_28_cc2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                            elif True:
                                d_29_ag_: _dafny.Seq
                                d_30_ai_: bool
                                d_31_ac_: _dafny.Seq
                                out22_: _dafny.Seq
                                out23_: bool
                                out24_: _dafny.Seq
                                out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                d_29_ag_ = out22_
                                d_30_ai_ = out23_
                                d_31_ac_ = out24_
                                generated = d_29_ag_
                                insideConstrainedOut = d_30_ai_
                                currentConstrainedOut = d_31_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_32_cg_: _dafny.Seq
                d_33_ci_: bool
                d_34_cc_: _dafny.Seq
                out25_: _dafny.Seq
                out26_: bool
                out27_: _dafny.Seq
                out25_, out26_, out27_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_32_cg_ = out25_
                d_33_ci_ = out26_
                d_34_cc_ = out27_
                generated = d_32_cg_
                insideConstrainedOut = d_33_ci_
                currentConstrainedOut = d_34_cc_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_35_rg_: _dafny.Seq
                d_36_rc_: _dafny.Seq
                out28_: _dafny.Seq
                out29_: _dafny.Seq
                out28_, out29_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                d_35_rg_ = out28_
                d_36_rc_ = out29_
                generated = d_35_rg_
                currentConstrainedOut = d_36_rc_
                if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                    d_37_cg2_: _dafny.Seq
                    d_38_ci2_: bool
                    d_39_cc2_: _dafny.Seq
                    out30_: _dafny.Seq
                    out31_: bool
                    out32_: _dafny.Seq
                    out30_, out31_, out32_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_37_cg2_ = out30_
                    d_38_ci2_ = out31_
                    d_39_cc2_ = out32_
                    generated = d_37_cg2_
                    insideConstrainedOut = d_38_ci2_
                    currentConstrainedOut = d_39_cc2_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    insideConstrainedOut = False
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        if ((not(insideConstrainedOut)) and (((d_1_steps_) + (3)) <= (maxSteps))) and ((d_1_steps_) < (maxSteps)):
            d_40_budgetRemaining_: int
            d_40_budgetRemaining_ = (maxSteps) - (d_1_steps_)
            if (d_40_budgetRemaining_) >= (3):
                d_41_og_: _dafny.Seq
                d_42_oi_: bool
                d_43_oc_: _dafny.Seq
                out33_: _dafny.Seq
                out34_: bool
                out35_: _dafny.Seq
                out33_, out34_, out35_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_41_og_ = out33_
                d_42_oi_ = out34_
                d_43_oc_ = out35_
                generated = d_41_og_
                insideConstrainedOut = d_42_oi_
                currentConstrainedOut = d_43_oc_
                d_1_steps_ = (d_1_steps_) + (1)
                with _dafny.label("2_0_0"):
                    while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                        with _dafny.c_label("2_0_0"):
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_44_cg_: _dafny.Seq
                                d_45_ci_: bool
                                d_46_cc_: _dafny.Seq
                                out36_: _dafny.Seq
                                out37_: bool
                                out38_: _dafny.Seq
                                out36_, out37_, out38_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_44_cg_ = out36_
                                d_45_ci_ = out37_
                                d_46_cc_ = out38_
                                generated = d_44_cg_
                                insideConstrainedOut = d_45_ci_
                                currentConstrainedOut = d_46_cc_
                                d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("2_0_0")
                            elif True:
                                d_47_narrow2_: bool
                                out39_: bool
                                out39_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                                d_47_narrow2_ = out39_
                                if d_47_narrow2_:
                                    d_48_rg2_: _dafny.Seq
                                    d_49_rc2_: _dafny.Seq
                                    out40_: _dafny.Seq
                                    out41_: _dafny.Seq
                                    out40_, out41_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                    d_48_rg2_ = out40_
                                    d_49_rc2_ = out41_
                                    generated = d_48_rg2_
                                    currentConstrainedOut = d_49_rc2_
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                        d_50_cg3_: _dafny.Seq
                                        d_51_ci3_: bool
                                        d_52_cc3_: _dafny.Seq
                                        out42_: _dafny.Seq
                                        out43_: bool
                                        out44_: _dafny.Seq
                                        out42_, out43_, out44_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_50_cg3_ = out42_
                                        d_51_ci3_ = out43_
                                        d_52_cc3_ = out44_
                                        generated = d_50_cg3_
                                        insideConstrainedOut = d_51_ci3_
                                        currentConstrainedOut = d_52_cc3_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        insideConstrainedOut = False
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    raise _dafny.Break("2_0_0")
                                elif True:
                                    d_53_constrainedPrompt2_: _dafny.Seq
                                    d_53_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_54_next2_: _dafny.Seq
                                    out45_: _dafny.Seq
                                    out45_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_53_constrainedPrompt2_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_5_penaltyTokens_, _dafny.BigRational('3e0'), 12, eosToken)
                                    d_54_next2_ = out45_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_54_next2_) == (eosToken):
                                        d_55_rg3_: _dafny.Seq
                                        d_56_rc3_: _dafny.Seq
                                        out46_: _dafny.Seq
                                        out47_: _dafny.Seq
                                        out46_, out47_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                        d_55_rg3_ = out46_
                                        d_56_rc3_ = out47_
                                        generated = d_55_rg3_
                                        currentConstrainedOut = d_56_rc3_
                                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                            d_57_cg4_: _dafny.Seq
                                            d_58_ci4_: bool
                                            d_59_cc4_: _dafny.Seq
                                            out48_: _dafny.Seq
                                            out49_: bool
                                            out50_: _dafny.Seq
                                            out48_, out49_, out50_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_57_cg4_ = out48_
                                            d_58_ci4_ = out49_
                                            d_59_cc4_ = out50_
                                            generated = d_57_cg4_
                                            insideConstrainedOut = d_58_ci4_
                                            currentConstrainedOut = d_59_cc4_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                        elif True:
                                            insideConstrainedOut = False
                                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                        raise _dafny.Break("2_0_0")
                                    elif True:
                                        d_60_ag2_: _dafny.Seq
                                        d_61_ai2_: bool
                                        d_62_ac2_: _dafny.Seq
                                        out51_: _dafny.Seq
                                        out52_: bool
                                        out53_: _dafny.Seq
                                        out51_, out52_, out53_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_54_next2_)
                                        d_60_ag2_ = out51_
                                        d_61_ai2_ = out52_
                                        d_62_ac2_ = out53_
                                        generated = d_60_ag2_
                                        insideConstrainedOut = d_61_ai2_
                                        currentConstrainedOut = d_62_ac2_
                            pass
                    pass
                if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                    if (parser).IsCompletePrefix(currentConstrainedOut):
                        d_63_cg5_: _dafny.Seq
                        d_64_ci5_: bool
                        d_65_cc5_: _dafny.Seq
                        out54_: _dafny.Seq
                        out55_: bool
                        out56_: _dafny.Seq
                        out54_, out55_, out56_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_63_cg5_ = out54_
                        d_64_ci5_ = out55_
                        d_65_cc5_ = out56_
                        generated = d_63_cg5_
                        insideConstrainedOut = d_64_ci5_
                        currentConstrainedOut = d_65_cc5_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_66_rg5_: _dafny.Seq
                        d_67_rc5_: _dafny.Seq
                        out57_: _dafny.Seq
                        out58_: _dafny.Seq
                        out57_, out58_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_66_rg5_ = out57_
                        d_67_rc5_ = out58_
                        generated = d_66_rg5_
                        currentConstrainedOut = d_67_rc5_
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                            d_68_cg6_: _dafny.Seq
                            d_69_ci6_: bool
                            d_70_cc6_: _dafny.Seq
                            out59_: _dafny.Seq
                            out60_: bool
                            out61_: _dafny.Seq
                            out59_, out60_, out61_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_68_cg6_ = out59_
                            d_69_ci6_ = out60_
                            d_70_cc6_ = out61_
                            generated = d_68_cg6_
                            insideConstrainedOut = d_69_ci6_
                            currentConstrainedOut = d_70_cc6_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

