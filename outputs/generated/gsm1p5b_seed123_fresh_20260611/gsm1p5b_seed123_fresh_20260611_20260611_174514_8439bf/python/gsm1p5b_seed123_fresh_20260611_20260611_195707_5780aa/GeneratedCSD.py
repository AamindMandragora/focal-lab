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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step in plain text. Write ALL reasoning outside << >>. Place ONLY the final numeric answer inside << >> using only numbers, +, -, *, /, //, **, (, ), and variable names. No int(), no ^, no curly braces. Example: <<n * m + k>>. Only ONE << >> span for the final answer.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanSteps_: int
        d_2_spanSteps_ = 0
        d_3_spanOpened_: bool
        d_3_spanOpened_ = insideConstrained
        d_4_maxSpanSteps_: int
        d_4_maxSpanSteps_ = 40
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_3_spanOpened_:
                            d_5_next1_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_5_next1_ = out0_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_5_next1_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next1_]))
                                if (d_5_next1_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_6_eg1_: _dafny.Seq
                                    d_7_ei1_: bool
                                    d_8_ec1_: _dafny.Seq
                                    out1_: _dafny.Seq
                                    out2_: bool
                                    out3_: _dafny.Seq
                                    out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_6_eg1_ = out1_
                                    d_7_ei1_ = out2_
                                    d_8_ec1_ = out3_
                                    generated = d_6_eg1_
                                    insideConstrainedOut = d_7_ei1_
                                    currentConstrainedOut = d_8_ec1_
                                    d_2_spanSteps_ = 0
                        elif True:
                            d_9_remaining_: int
                            d_9_remaining_ = (maxSteps) - (d_1_steps_)
                            d_10_reserve_: int
                            d_10_reserve_ = 15
                            if (d_9_remaining_) <= (d_10_reserve_):
                                if (d_9_remaining_) >= (2):
                                    d_11_og2_: _dafny.Seq
                                    d_12_oi2_: bool
                                    d_13_oc2_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_11_og2_ = out4_
                                    d_12_oi2_ = out5_
                                    d_13_oc2_ = out6_
                                    generated = d_11_og2_
                                    insideConstrainedOut = d_12_oi2_
                                    currentConstrainedOut = d_13_oc2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_2_spanSteps_ = 0
                                    d_3_spanOpened_ = True
                                elif True:
                                    d_14_next2_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out7_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                    d_14_next2_ = out7_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_14_next2_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_14_next2_]))
                                        if (d_14_next2_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                            d_15_eg2_: _dafny.Seq
                                            d_16_ei2_: bool
                                            d_17_ec2_: _dafny.Seq
                                            out8_: _dafny.Seq
                                            out9_: bool
                                            out10_: _dafny.Seq
                                            out8_, out9_, out10_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                            d_15_eg2_ = out8_
                                            d_16_ei2_ = out9_
                                            d_17_ec2_ = out10_
                                            generated = d_15_eg2_
                                            insideConstrainedOut = d_16_ei2_
                                            currentConstrainedOut = d_17_ec2_
                                            d_2_spanSteps_ = 0
                                            d_3_spanOpened_ = True
                            elif True:
                                d_18_chunkBudget_: int
                                if ((d_9_remaining_) - (d_10_reserve_)) < (60):
                                    d_18_chunkBudget_ = (d_9_remaining_) - (d_10_reserve_)
                                elif True:
                                    d_18_chunkBudget_ = 60
                                if (d_18_chunkBudget_) == (0):
                                    d_18_chunkBudget_ = 1
                                d_19_chunkGenerated_: _dafny.Seq
                                d_20_stoppedOnOpenSpan_: bool
                                d_21_stoppedOnEos_: bool
                                d_22_stepsUsed_: int
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: bool
                                out14_: int
                                out11_, out12_, out13_, out14_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_18_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                                d_19_chunkGenerated_ = out11_
                                d_20_stoppedOnOpenSpan_ = out12_
                                d_21_stoppedOnEos_ = out13_
                                d_22_stepsUsed_ = out14_
                                d_1_steps_ = (d_1_steps_) + (d_22_stepsUsed_)
                                generated = d_19_chunkGenerated_
                                if d_20_stoppedOnOpenSpan_:
                                    d_23_eg3_: _dafny.Seq
                                    d_24_ei3_: bool
                                    d_25_ec3_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_23_eg3_ = out15_
                                    d_24_ei3_ = out16_
                                    d_25_ec3_ = out17_
                                    generated = d_23_eg3_
                                    insideConstrainedOut = d_24_ei3_
                                    currentConstrainedOut = d_25_ec3_
                                    d_2_spanSteps_ = 0
                                    d_3_spanOpened_ = True
                                elif d_21_stoppedOnEos_:
                                    raise _dafny.Break("0")
                    elif True:
                        d_26_remaining4_: int
                        d_26_remaining4_ = (maxSteps) - (d_1_steps_)
                        d_27_shouldForceClose_: bool
                        d_27_shouldForceClose_ = ((d_2_spanSteps_) >= (d_4_maxSpanSteps_)) or ((d_26_remaining4_) <= (1))
                        if d_27_shouldForceClose_:
                            d_28_rg4_: _dafny.Seq
                            d_29_rc4_: _dafny.Seq
                            out18_: _dafny.Seq
                            out19_: _dafny.Seq
                            out18_, out19_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_28_rg4_ = out18_
                            d_29_rc4_ = out19_
                            generated = d_28_rg4_
                            currentConstrainedOut = d_29_rc4_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_30_cG4_: _dafny.Seq
                                d_31_cI4_: bool
                                d_32_cC4_: _dafny.Seq
                                out20_: _dafny.Seq
                                out21_: bool
                                out22_: _dafny.Seq
                                out20_, out21_, out22_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_30_cG4_ = out20_
                                d_31_cI4_ = out21_
                                d_32_cC4_ = out22_
                                generated = d_30_cG4_
                                insideConstrainedOut = d_31_cI4_
                                currentConstrainedOut = d_32_cC4_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_spanSteps_ = 0
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_spanSteps_ = 0
                                if (d_1_steps_) < (maxSteps):
                                    d_33_dummy4_: _dafny.Seq
                                    out23_: _dafny.Seq
                                    out23_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                    d_33_dummy4_ = out23_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_33_dummy4_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_33_dummy4_]))
                                        if (d_33_dummy4_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                            d_34_eg4_: _dafny.Seq
                                            d_35_ei4_: bool
                                            d_36_ec4_: _dafny.Seq
                                            out24_: _dafny.Seq
                                            out25_: bool
                                            out26_: _dafny.Seq
                                            out24_, out25_, out26_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                            d_34_eg4_ = out24_
                                            d_35_ei4_ = out25_
                                            d_36_ec4_ = out26_
                                            generated = d_34_eg4_
                                            insideConstrainedOut = d_35_ei4_
                                            currentConstrainedOut = d_36_ec4_
                                            d_2_spanSteps_ = 0
                        elif True:
                            d_37_cg5_: _dafny.Seq
                            d_38_ci5_: bool
                            d_39_cc5_: _dafny.Seq
                            d_40_closed5_: bool
                            out27_: _dafny.Seq
                            out28_: bool
                            out29_: _dafny.Seq
                            out30_: bool
                            out27_, out28_, out29_, out30_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_37_cg5_ = out27_
                            d_38_ci5_ = out28_
                            d_39_cc5_ = out29_
                            d_40_closed5_ = out30_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if d_40_closed5_:
                                generated = d_37_cg5_
                                insideConstrainedOut = d_38_ci5_
                                currentConstrainedOut = d_39_cc5_
                                d_2_spanSteps_ = 0
                            elif True:
                                if (d_1_steps_) < (maxSteps):
                                    d_41_constrainedPrompt5_: _dafny.Seq
                                    d_41_constrainedPrompt5_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_42_next5_: _dafny.Seq
                                    out31_: _dafny.Seq
                                    out31_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_41_constrainedPrompt5_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                    d_42_next5_ = out31_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_42_next5_) == (eosToken):
                                        d_43_rg5_: _dafny.Seq
                                        d_44_rc5_: _dafny.Seq
                                        out32_: _dafny.Seq
                                        out33_: _dafny.Seq
                                        out32_, out33_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                        d_43_rg5_ = out32_
                                        d_44_rc5_ = out33_
                                        generated = d_43_rg5_
                                        currentConstrainedOut = d_44_rc5_
                                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                            d_45_cG5_: _dafny.Seq
                                            d_46_cI5_: bool
                                            d_47_cC5_: _dafny.Seq
                                            out34_: _dafny.Seq
                                            out35_: bool
                                            out36_: _dafny.Seq
                                            out34_, out35_, out36_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_45_cG5_ = out34_
                                            d_46_cI5_ = out35_
                                            d_47_cC5_ = out36_
                                            generated = d_45_cG5_
                                            insideConstrainedOut = d_46_cI5_
                                            currentConstrainedOut = d_47_cC5_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                            d_2_spanSteps_ = 0
                                        elif True:
                                            insideConstrainedOut = False
                                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                            d_2_spanSteps_ = 0
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_48_isComplete5_: bool
                                        d_48_isComplete5_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                        if not(d_48_isComplete5_):
                                            d_49_ag5_: _dafny.Seq
                                            d_50_ai5_: bool
                                            d_51_ac5_: _dafny.Seq
                                            out37_: _dafny.Seq
                                            out38_: bool
                                            out39_: _dafny.Seq
                                            out37_, out38_, out39_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_42_next5_)
                                            d_49_ag5_ = out37_
                                            d_50_ai5_ = out38_
                                            d_51_ac5_ = out39_
                                            generated = d_49_ag5_
                                            insideConstrainedOut = d_50_ai5_
                                            currentConstrainedOut = d_51_ac5_
                                            d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                    pass
            pass
        if insideConstrainedOut:
            d_52_rg6_: _dafny.Seq
            d_53_rc6_: _dafny.Seq
            out40_: _dafny.Seq
            out41_: _dafny.Seq
            out40_, out41_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_52_rg6_ = out40_
            d_53_rc6_ = out41_
            generated = d_52_rg6_
            currentConstrainedOut = d_53_rc6_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_54_cG6_: _dafny.Seq
                d_55_cI6_: bool
                d_56_cC6_: _dafny.Seq
                out42_: _dafny.Seq
                out43_: bool
                out44_: _dafny.Seq
                out42_, out43_, out44_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_54_cG6_ = out42_
                d_55_cI6_ = out43_
                d_56_cC6_ = out44_
                generated = d_54_cG6_
                insideConstrainedOut = d_55_cI6_
                currentConstrainedOut = d_56_cC6_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        if ((not(d_3_spanOpened_)) and (not(insideConstrainedOut))) and (((d_1_steps_) + (2)) <= (maxSteps)):
            d_57_og7_: _dafny.Seq
            d_58_oi7_: bool
            d_59_oc7_: _dafny.Seq
            out45_: _dafny.Seq
            out46_: bool
            out47_: _dafny.Seq
            out45_, out46_, out47_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_57_og7_ = out45_
            d_58_oi7_ = out46_
            d_59_oc7_ = out47_
            generated = d_57_og7_
            insideConstrainedOut = d_58_oi7_
            currentConstrainedOut = d_59_oc7_
            d_1_steps_ = (d_1_steps_) + (1)
            if (d_1_steps_) < (maxSteps):
                d_60_constrainedPrompt7_: _dafny.Seq
                d_60_constrainedPrompt7_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_61_next7_: _dafny.Seq
                out48_: _dafny.Seq
                out48_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_60_constrainedPrompt7_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                d_61_next7_ = out48_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_61_next7_) != (eosToken):
                    d_62_isComplete7_: bool
                    d_62_isComplete7_ = (parser).IsCompletePrefix(currentConstrainedOut)
                    if not(d_62_isComplete7_):
                        d_63_ag7_: _dafny.Seq
                        d_64_ai7_: bool
                        d_65_ac7_: _dafny.Seq
                        out49_: _dafny.Seq
                        out50_: bool
                        out51_: _dafny.Seq
                        out49_, out50_, out51_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_61_next7_)
                        d_63_ag7_ = out49_
                        d_64_ai7_ = out50_
                        d_65_ac7_ = out51_
                        generated = d_63_ag7_
                        insideConstrainedOut = d_64_ai7_
                        currentConstrainedOut = d_65_ac7_
                    if (d_1_steps_) < (maxSteps):
                        d_66_rg7_: _dafny.Seq
                        d_67_rc7_: _dafny.Seq
                        out52_: _dafny.Seq
                        out53_: _dafny.Seq
                        out52_, out53_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_66_rg7_ = out52_
                        d_67_rc7_ = out53_
                        generated = d_66_rg7_
                        currentConstrainedOut = d_67_rc7_
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_68_cG7_: _dafny.Seq
                            d_69_cI7_: bool
                            d_70_cC7_: _dafny.Seq
                            out54_: _dafny.Seq
                            out55_: bool
                            out56_: _dafny.Seq
                            out54_, out55_, out56_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_68_cG7_ = out54_
                            d_69_cI7_ = out55_
                            d_70_cC7_ = out56_
                            generated = d_68_cG7_
                            insideConstrainedOut = d_69_cI7_
                            currentConstrainedOut = d_70_cC7_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        insideConstrainedOut = False
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                elif True:
                    d_71_rg7b_: _dafny.Seq
                    d_72_rc7b_: _dafny.Seq
                    out57_: _dafny.Seq
                    out58_: _dafny.Seq
                    out57_, out58_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                    d_71_rg7b_ = out57_
                    d_72_rc7b_ = out58_
                    generated = d_71_rg7b_
                    currentConstrainedOut = d_72_rc7b_
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                        d_73_cG7b_: _dafny.Seq
                        d_74_cI7b_: bool
                        d_75_cC7b_: _dafny.Seq
                        out59_: _dafny.Seq
                        out60_: bool
                        out61_: _dafny.Seq
                        out59_, out60_, out61_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_73_cG7b_ = out59_
                        d_74_cI7b_ = out60_
                        d_75_cC7b_ = out61_
                        generated = d_73_cG7b_
                        insideConstrainedOut = d_74_cI7b_
                        currentConstrainedOut = d_75_cC7b_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        insideConstrainedOut = False
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            elif True:
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

