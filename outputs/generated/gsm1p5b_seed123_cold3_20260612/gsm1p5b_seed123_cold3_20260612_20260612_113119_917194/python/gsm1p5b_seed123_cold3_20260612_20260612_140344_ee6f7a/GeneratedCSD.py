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
        (d_0_helpers_).AppendTaskGuidance(lm, (((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Show each calculation inside << >> delimiters using simple arithmetic only. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use only integers, variable names, +, -, *, /, (, ) inside << >>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Example: Total is <<n * p>>. The answer is <<n * p>>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write the final answer as <<expression>> at the end. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Keep << >> expressions short and close them immediately after the expression."))))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_closeReserve_: int
        d_2_closeReserve_ = 15
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_3_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                            d_4_og_: _dafny.Seq
                            d_5_oi_: bool
                            d_6_oc_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_4_og_ = out1_
                            d_5_oi_ = out2_
                            d_6_oc_ = out3_
                            generated = d_4_og_
                            insideConstrainedOut = d_5_oi_
                            currentConstrainedOut = d_6_oc_
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                    elif (insideConstrainedOut) and (((d_1_steps_) + (d_2_closeReserve_)) >= (maxSteps)):
                        d_7_rg_: _dafny.Seq
                        d_8_rc_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: _dafny.Seq
                        out4_, out5_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_7_rg_ = out4_
                        d_8_rc_ = out5_
                        generated = d_7_rg_
                        currentConstrainedOut = d_8_rc_
                        d_9_isComplete_: bool
                        d_9_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if (d_9_isComplete_) and ((d_1_steps_) < (maxSteps)):
                            d_10_fg_: _dafny.Seq
                            d_11_fi_: bool
                            d_12_fc_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_10_fg_ = out6_
                            d_11_fi_ = out7_
                            d_12_fc_ = out8_
                            d_1_steps_ = (d_1_steps_) + (1)
                            generated = d_10_fg_
                            insideConstrainedOut = d_11_fi_
                            currentConstrainedOut = d_12_fc_
                        elif True:
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_13_cg_: _dafny.Seq
                        d_14_ci_: bool
                        d_15_cc_: _dafny.Seq
                        d_16_closed_: bool
                        out9_: _dafny.Seq
                        out10_: bool
                        out11_: _dafny.Seq
                        out12_: bool
                        out9_, out10_, out11_, out12_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_13_cg_ = out9_
                        d_14_ci_ = out10_
                        d_15_cc_ = out11_
                        d_16_closed_ = out12_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_16_closed_:
                            generated = d_13_cg_
                            insideConstrainedOut = d_14_ci_
                            currentConstrainedOut = d_15_cc_
                        elif True:
                            d_17_constrainedPrompt_: _dafny.Seq
                            d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_18_next_: _dafny.Seq
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_18_next_ = out13_
                            if (d_18_next_) == (eosToken):
                                d_19_rg_: _dafny.Seq
                                d_20_rc_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: _dafny.Seq
                                out14_, out15_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_19_rg_ = out14_
                                d_20_rc_ = out15_
                                generated = d_19_rg_
                                currentConstrainedOut = d_20_rc_
                                d_21_isComplete_: bool
                                d_21_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if (d_21_isComplete_) and ((d_1_steps_) < (maxSteps)):
                                    d_22_fg_: _dafny.Seq
                                    d_23_fi_: bool
                                    d_24_fc_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_22_fg_ = out16_
                                    d_23_fi_ = out17_
                                    d_24_fc_ = out18_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    generated = d_22_fg_
                                    insideConstrainedOut = d_23_fi_
                                    currentConstrainedOut = d_24_fc_
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                            elif True:
                                d_25_ag_: _dafny.Seq
                                d_26_ai_: bool
                                d_27_ac_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                                d_25_ag_ = out19_
                                d_26_ai_ = out20_
                                d_27_ac_ = out21_
                                generated = d_25_ag_
                                insideConstrainedOut = d_26_ai_
                                currentConstrainedOut = d_27_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_28_rg_: _dafny.Seq
            d_29_rc_: _dafny.Seq
            out22_: _dafny.Seq
            out23_: _dafny.Seq
            out22_, out23_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_28_rg_ = out22_
            d_29_rc_ = out23_
            generated = d_28_rg_
            currentConstrainedOut = d_29_rc_
            d_30_isComplete_: bool
            d_30_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if d_30_isComplete_:
                d_31_fg_: _dafny.Seq
                d_32_fi_: bool
                d_33_fc_: _dafny.Seq
                out24_: _dafny.Seq
                out25_: bool
                out26_: _dafny.Seq
                out24_, out25_, out26_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_31_fg_ = out24_
                d_32_fi_ = out25_
                d_33_fc_ = out26_
                d_1_steps_ = (d_1_steps_) + (1)
                generated = d_31_fg_
                insideConstrainedOut = d_32_fi_
                currentConstrainedOut = d_33_fc_
            elif True:
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

