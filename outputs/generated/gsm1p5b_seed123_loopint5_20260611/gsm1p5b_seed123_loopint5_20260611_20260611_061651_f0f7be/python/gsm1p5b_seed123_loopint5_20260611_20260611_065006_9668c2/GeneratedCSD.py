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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "After reasoning step by step, provide the final arithmetic expression as the answer inside << >> delimiters. The expression should use the symbolic variable names from the problem.")))
        d_1_SPAN__RESERVE_: int
        d_1_SPAN__RESERVE_ = 150
        d_2_steps_: int
        d_2_steps_ = 0
        (d_0_helpers_).SafePenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('5e1'))
        d_3_reasoningBudget_: int
        d_3_reasoningBudget_ = 0
        if (maxSteps) > ((d_1_SPAN__RESERVE_) + (1)):
            d_3_reasoningBudget_ = ((maxSteps) - (d_1_SPAN__RESERVE_)) - (1)
        d_4_reasoningSteps_: int
        d_4_reasoningSteps_ = 0
        with _dafny.label("0"):
            while ((d_4_reasoningSteps_) < (d_3_reasoningBudget_)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    (d_0_helpers_).SafePenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('5e1'))
                    d_5_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_5_next_ = out0_
                    d_4_reasoningSteps_ = (d_4_reasoningSteps_) + (1)
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_5_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        d_6_g2_: _dafny.Seq
                        d_7_i2_: bool
                        d_8_c2_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_6_g2_ = out1_
                        d_7_i2_ = out2_
                        d_8_c2_ = out3_
                        generated = d_6_g2_
                        insideConstrainedOut = d_7_i2_
                        currentConstrainedOut = d_8_c2_
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                    pass
            pass
        if ((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_9_g2_: _dafny.Seq
            d_10_i2_: bool
            d_11_c2_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_9_g2_ = out4_
            d_10_i2_ = out5_
            d_11_c2_ = out6_
            generated = d_9_g2_
            insideConstrainedOut = d_10_i2_
            currentConstrainedOut = d_11_c2_
            d_2_steps_ = (d_2_steps_) + (1)
        with _dafny.label("1"):
            while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    if (parser).IsCompletePrefix(currentConstrainedOut):
                        d_12_cg_: _dafny.Seq
                        d_13_ci_: bool
                        d_14_cc_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_12_cg_ = out7_
                        d_13_ci_ = out8_
                        d_14_cc_ = out9_
                        generated = d_12_cg_
                        insideConstrainedOut = d_13_ci_
                        currentConstrainedOut = d_14_cc_
                        d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("1")
                    elif True:
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_16_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_16_next_ = out10_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_16_next_) == (eosToken):
                            d_17_rg_: _dafny.Seq
                            d_18_rc_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: _dafny.Seq
                            out11_, out12_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_17_rg_ = out11_
                            d_18_rc_ = out12_
                            generated = d_17_rg_
                            currentConstrainedOut = d_18_rc_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
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
                                d_2_steps_ = (d_2_steps_) + (1)
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            raise _dafny.Break("1")
                        elif True:
                            d_22_ag_: _dafny.Seq
                            d_23_ai_: bool
                            d_24_ac_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                            d_22_ag_ = out16_
                            d_23_ai_ = out17_
                            d_24_ac_ = out18_
                            generated = d_22_ag_
                            insideConstrainedOut = d_23_ai_
                            currentConstrainedOut = d_24_ac_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                d_25_cg3_: _dafny.Seq
                                d_26_ci3_: bool
                                d_27_cc3_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_25_cg3_ = out19_
                                d_26_ci3_ = out20_
                                d_27_cc3_ = out21_
                                generated = d_25_cg3_
                                insideConstrainedOut = d_26_ci3_
                                currentConstrainedOut = d_27_cc3_
                                d_2_steps_ = (d_2_steps_) + (1)
                                raise _dafny.Break("1")
                    pass
            pass
        if insideConstrainedOut:
            d_28_rg_: _dafny.Seq
            d_29_rc_: _dafny.Seq
            out22_: _dafny.Seq
            out23_: _dafny.Seq
            out22_, out23_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_28_rg_ = out22_
            d_29_rc_ = out23_
            generated = d_28_rg_
            currentConstrainedOut = d_29_rc_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                d_30_cg_: _dafny.Seq
                d_31_ci_: bool
                d_32_cc_: _dafny.Seq
                out24_: _dafny.Seq
                out25_: bool
                out26_: _dafny.Seq
                out24_, out25_, out26_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_30_cg_ = out24_
                d_31_ci_ = out25_
                d_32_cc_ = out26_
                generated = d_30_cg_
                insideConstrainedOut = d_31_ci_
                currentConstrainedOut = d_32_cc_
                d_2_steps_ = (d_2_steps_) + (1)
            elif True:
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

