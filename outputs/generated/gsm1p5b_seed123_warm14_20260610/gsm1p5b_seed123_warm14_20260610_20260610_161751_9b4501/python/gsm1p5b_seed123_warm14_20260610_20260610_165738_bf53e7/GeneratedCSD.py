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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. At the very end write the final answer as an arithmetic expression inside << >>. Use only: integers, +, -, *, /, (, ).")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanReserve_: int
        if (maxSteps) >= (60):
            d_2_spanReserve_ = 30
        elif (maxSteps) >= (20):
            d_2_spanReserve_ = 15
        elif (maxSteps) >= (4):
            d_2_spanReserve_ = 4
        elif True:
            d_2_spanReserve_ = maxSteps
        d_3_freeLimit_: int
        if (maxSteps) > (d_2_spanReserve_):
            d_3_freeLimit_ = (maxSteps) - (d_2_spanReserve_)
        elif True:
            d_3_freeLimit_ = 0
        d_4_penaltyToks_: _dafny.Seq
        d_4_penaltyToks_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "{")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "}")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "%")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "#")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "@")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "$")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "^")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "&")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "|")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "~"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_1_steps_) >= (d_3_freeLimit_):
                            d_5_og_: _dafny.Seq
                            d_6_oi_: bool
                            d_7_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_og_ = out0_
                            d_6_oi_ = out1_
                            d_7_oc_ = out2_
                            generated = d_5_og_
                            insideConstrainedOut = d_6_oi_
                            currentConstrainedOut = d_7_oc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                    elif True:
                        d_9_cg_: _dafny.Seq
                        d_10_ci_: bool
                        d_11_cc_: _dafny.Seq
                        d_12_closed_: bool
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out7_: bool
                        out4_, out5_, out6_, out7_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_9_cg_ = out4_
                        d_10_ci_ = out5_
                        d_11_cc_ = out6_
                        d_12_closed_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_12_closed_:
                            generated = d_9_cg_
                            insideConstrainedOut = d_10_ci_
                            currentConstrainedOut = d_11_cc_
                            raise _dafny.Break("0")
                        elif True:
                            if (d_1_steps_) < (maxSteps):
                                d_13_constrainedPrompt_: _dafny.Seq
                                d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_14_next_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, d_4_penaltyToks_, _dafny.BigRational('8e0'), eosToken)
                                d_14_next_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_14_next_) == (eosToken):
                                    d_15_rg_: _dafny.Seq
                                    d_16_rc_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out9_, out10_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                    d_15_rg_ = out9_
                                    d_16_rc_ = out10_
                                    generated = d_15_rg_
                                    currentConstrainedOut = d_16_rc_
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_17_cg2_: _dafny.Seq
                                        d_18_ci2_: bool
                                        d_19_cc2_: _dafny.Seq
                                        out11_: _dafny.Seq
                                        out12_: bool
                                        out13_: _dafny.Seq
                                        out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_17_cg2_ = out11_
                                        d_18_ci2_ = out12_
                                        d_19_cc2_ = out13_
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
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                    d_20_ag_ = out14_
                                    d_21_ai_ = out15_
                                    d_22_ac_ = out16_
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
                out17_: _dafny.Seq
                out18_: bool
                out19_: _dafny.Seq
                out17_, out18_, out19_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_23_cg_ = out17_
                d_24_ci_ = out18_
                d_25_cc_ = out19_
                generated = d_23_cg_
                insideConstrainedOut = d_24_ci_
                currentConstrainedOut = d_25_cc_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_26_rg_: _dafny.Seq
                d_27_rc_: _dafny.Seq
                out20_: _dafny.Seq
                out21_: _dafny.Seq
                out20_, out21_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                d_26_rg_ = out20_
                d_27_rc_ = out21_
                generated = d_26_rg_
                currentConstrainedOut = d_27_rc_
                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                    d_28_cg2_: _dafny.Seq
                    d_29_ci2_: bool
                    d_30_cc2_: _dafny.Seq
                    out22_: _dafny.Seq
                    out23_: bool
                    out24_: _dafny.Seq
                    out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_28_cg2_ = out22_
                    d_29_ci2_ = out23_
                    d_30_cc2_ = out24_
                    generated = d_28_cg2_
                    insideConstrainedOut = d_29_ci2_
                    currentConstrainedOut = d_30_cc2_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    insideConstrainedOut = False
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

