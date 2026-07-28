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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. At the very end, write ONLY the final arithmetic expression inside << >> delimiters. Use only variable names, numbers, +, -, *, /, (, ) inside the delimiters. Example: The answer is <<n * (mult + 1)>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freePhaseLimit_: int
        if (maxSteps) >= (100):
            d_2_freePhaseLimit_ = 80
        elif (maxSteps) >= (20):
            d_2_freePhaseLimit_ = _dafny.euclidian_division(maxSteps, 5)
        elif (maxSteps) >= (4):
            d_2_freePhaseLimit_ = (maxSteps) - (3)
        elif True:
            d_2_freePhaseLimit_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_1_steps_) >= (d_2_freePhaseLimit_):
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
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_6_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_6_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_6_next_) == (eosToken):
                                if ((d_1_steps_) + (2)) <= (maxSteps):
                                    d_7_og_: _dafny.Seq
                                    d_8_oi_: bool
                                    d_9_oc_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_7_og_ = out4_
                                    d_8_oi_ = out5_
                                    d_9_oc_ = out6_
                                    generated = d_7_og_
                                    insideConstrainedOut = d_8_oi_
                                    currentConstrainedOut = d_9_oc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_10_cg_: _dafny.Seq
                        d_11_ci_: bool
                        d_12_cc_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_10_cg_ = out7_
                        d_11_ci_ = out8_
                        d_12_cc_ = out9_
                        generated = d_10_cg_
                        insideConstrainedOut = d_11_ci_
                        currentConstrainedOut = d_12_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_13_constrainedPrompt_: _dafny.Seq
                        d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_14_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                        d_14_next_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_14_next_) == (eosToken):
                            d_15_rg_: _dafny.Seq
                            d_16_rc_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: _dafny.Seq
                            out11_, out12_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_15_rg_ = out11_
                            d_16_rc_ = out12_
                            generated = d_15_rg_
                            currentConstrainedOut = d_16_rc_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                d_17_cg2_: _dafny.Seq
                                d_18_ci2_: bool
                                d_19_cc2_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_17_cg2_ = out13_
                                d_18_ci2_ = out14_
                                d_19_cc2_ = out15_
                                generated = d_17_cg2_
                                insideConstrainedOut = d_18_ci2_
                                currentConstrainedOut = d_19_cc2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            raise _dafny.Break("0")
                        elif True:
                            d_20_ag_: _dafny.Seq
                            d_21_ai_: bool
                            d_22_ac_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                            d_20_ag_ = out16_
                            d_21_ai_ = out17_
                            d_22_ac_ = out18_
                            generated = d_20_ag_
                            insideConstrainedOut = d_21_ai_
                            currentConstrainedOut = d_22_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_23_cg_: _dafny.Seq
                d_24_ci_: bool
                d_25_cc_: _dafny.Seq
                out19_: _dafny.Seq
                out20_: bool
                out21_: _dafny.Seq
                out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_23_cg_ = out19_
                d_24_ci_ = out20_
                d_25_cc_ = out21_
                generated = d_23_cg_
                insideConstrainedOut = d_24_ci_
                currentConstrainedOut = d_25_cc_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_26_rg_: _dafny.Seq
                d_27_rc_: _dafny.Seq
                out22_: _dafny.Seq
                out23_: _dafny.Seq
                out22_, out23_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                d_26_rg_ = out22_
                d_27_rc_ = out23_
                generated = d_26_rg_
                currentConstrainedOut = d_27_rc_
                if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                    d_28_cg2_: _dafny.Seq
                    d_29_ci2_: bool
                    d_30_cc2_: _dafny.Seq
                    out24_: _dafny.Seq
                    out25_: bool
                    out26_: _dafny.Seq
                    out24_, out25_, out26_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_28_cg2_ = out24_
                    d_29_ci2_ = out25_
                    d_30_cc2_ = out26_
                    generated = d_28_cg2_
                    insideConstrainedOut = d_29_ci2_
                    currentConstrainedOut = d_30_cc2_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    insideConstrainedOut = False
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

