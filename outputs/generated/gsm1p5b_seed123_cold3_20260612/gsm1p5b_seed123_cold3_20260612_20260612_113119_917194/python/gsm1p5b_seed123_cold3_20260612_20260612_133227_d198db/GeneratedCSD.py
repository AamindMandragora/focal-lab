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
        (d_0_helpers_).AppendTaskGuidance(lm, ((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write your reasoning, then for each calculation write <<expression>>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use only numbers, variable names, +, -, *, /, (, ) inside << >>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "No braces {}, no ** exponentiation, no function calls inside << >>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "The final answer must be the last <<expression>>."))))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_chunkBudget_: int
        d_2_chunkBudget_ = 35
        d_3_spanBudget_: int
        d_3_spanBudget_ = 40
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_remaining_: int
                        d_4_remaining_ = (maxSteps) - (d_1_steps_)
                        d_5_reserve_: int
                        d_5_reserve_ = (d_3_spanBudget_) + (2)
                        if (d_4_remaining_) <= (d_5_reserve_):
                            if (d_4_remaining_) == (0):
                                raise _dafny.Break("0")
                            d_6_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_6_next_ = out0_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_6_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                                d_7_og_: _dafny.Seq
                                d_8_oi_: bool
                                d_9_oc_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_7_og_ = out1_
                                d_8_oi_ = out2_
                                d_9_oc_ = out3_
                                generated = d_7_og_
                                insideConstrainedOut = d_8_oi_
                                currentConstrainedOut = d_9_oc_
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                        elif True:
                            d_10_actualChunk_: int
                            d_10_actualChunk_ = d_2_chunkBudget_
                            if (d_10_actualChunk_) > ((d_4_remaining_) - (d_5_reserve_)):
                                d_10_actualChunk_ = (d_4_remaining_) - (d_5_reserve_)
                            if (d_10_actualChunk_) == (0):
                                d_11_og_: _dafny.Seq
                                d_12_oi_: bool
                                d_13_oc_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_11_og_ = out4_
                                d_12_oi_ = out5_
                                d_13_oc_ = out6_
                                d_1_steps_ = (d_1_steps_) + (1)
                                generated = d_11_og_
                                insideConstrainedOut = d_12_oi_
                                currentConstrainedOut = d_13_oc_
                            elif True:
                                d_14_genOut_: _dafny.Seq
                                d_15_stoppedOnOpen_: bool
                                d_16_stoppedOnEos_: bool
                                d_17_stepsUsed_: int
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: bool
                                out10_: int
                                out7_, out8_, out9_, out10_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_10_actualChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                                d_14_genOut_ = out7_
                                d_15_stoppedOnOpen_ = out8_
                                d_16_stoppedOnEos_ = out9_
                                d_17_stepsUsed_ = out10_
                                d_1_steps_ = (d_1_steps_) + (d_17_stepsUsed_)
                                generated = d_14_genOut_
                                if d_16_stoppedOnEos_:
                                    raise _dafny.Break("0")
                                elif d_15_stoppedOnOpen_:
                                    d_18_og_: _dafny.Seq
                                    d_19_oi_: bool
                                    d_20_oc_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_18_og_ = out11_
                                    d_19_oi_ = out12_
                                    d_20_oc_ = out13_
                                    generated = d_18_og_
                                    insideConstrainedOut = d_19_oi_
                                    currentConstrainedOut = d_20_oc_
                                elif True:
                                    if (d_1_steps_) < (maxSteps):
                                        d_21_og_: _dafny.Seq
                                        d_22_oi_: bool
                                        d_23_oc_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out15_: bool
                                        out16_: _dafny.Seq
                                        out14_, out15_, out16_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                        d_21_og_ = out14_
                                        d_22_oi_ = out15_
                                        d_23_oc_ = out16_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        generated = d_21_og_
                                        insideConstrainedOut = d_22_oi_
                                        currentConstrainedOut = d_23_oc_
                    elif True:
                        d_24_cg_: _dafny.Seq
                        d_25_ci_: bool
                        d_26_cc_: _dafny.Seq
                        d_27_closed_: bool
                        out17_: _dafny.Seq
                        out18_: bool
                        out19_: _dafny.Seq
                        out20_: bool
                        out17_, out18_, out19_, out20_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_24_cg_ = out17_
                        d_25_ci_ = out18_
                        d_26_cc_ = out19_
                        d_27_closed_ = out20_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_27_closed_:
                            generated = d_24_cg_
                            insideConstrainedOut = d_25_ci_
                            currentConstrainedOut = d_26_cc_
                        elif True:
                            d_28_constrainedPrompt_: _dafny.Seq
                            d_28_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_29_next_: _dafny.Seq
                            out21_: _dafny.Seq
                            out21_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_28_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_29_next_ = out21_
                            if (d_29_next_) == (eosToken):
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                            elif True:
                                d_30_ag_: _dafny.Seq
                                d_31_ai_: bool
                                d_32_ac_: _dafny.Seq
                                out22_: _dafny.Seq
                                out23_: bool
                                out24_: _dafny.Seq
                                out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_29_next_)
                                d_30_ag_ = out22_
                                d_31_ai_ = out23_
                                d_32_ac_ = out24_
                                generated = d_30_ag_
                                insideConstrainedOut = d_31_ai_
                                currentConstrainedOut = d_32_ac_
                                if (len(currentConstrainedOut)) > (d_3_spanBudget_):
                                    d_33_rg_: _dafny.Seq
                                    d_34_rc_: _dafny.Seq
                                    out25_: _dafny.Seq
                                    out26_: _dafny.Seq
                                    out25_, out26_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                    d_33_rg_ = out25_
                                    d_34_rc_ = out26_
                                    generated = d_33_rg_
                                    currentConstrainedOut = d_34_rc_
                                    d_35_isComplete_: bool
                                    d_35_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if (d_35_isComplete_) and ((d_1_steps_) < (maxSteps)):
                                        d_36_fg_: _dafny.Seq
                                        d_37_fi_: bool
                                        d_38_fc_: _dafny.Seq
                                        out27_: _dafny.Seq
                                        out28_: bool
                                        out29_: _dafny.Seq
                                        out27_, out28_, out29_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_36_fg_ = out27_
                                        d_37_fi_ = out28_
                                        d_38_fc_ = out29_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        generated = d_36_fg_
                                        insideConstrainedOut = d_37_fi_
                                        currentConstrainedOut = d_38_fc_
                                    elif True:
                                        insideConstrainedOut = False
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_39_rg_: _dafny.Seq
            d_40_rc_: _dafny.Seq
            out30_: _dafny.Seq
            out31_: _dafny.Seq
            out30_, out31_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_39_rg_ = out30_
            d_40_rc_ = out31_
            generated = d_39_rg_
            currentConstrainedOut = d_40_rc_
            d_41_isComplete_: bool
            d_41_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if d_41_isComplete_:
                d_42_fg_: _dafny.Seq
                d_43_fi_: bool
                d_44_fc_: _dafny.Seq
                out32_: _dafny.Seq
                out33_: bool
                out34_: _dafny.Seq
                out32_, out33_, out34_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_42_fg_ = out32_
                d_43_fi_ = out33_
                d_44_fc_ = out34_
                d_1_steps_ = (d_1_steps_) + (1)
                generated = d_42_fg_
                insideConstrainedOut = d_43_fi_
                currentConstrainedOut = d_44_fc_
            elif True:
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

