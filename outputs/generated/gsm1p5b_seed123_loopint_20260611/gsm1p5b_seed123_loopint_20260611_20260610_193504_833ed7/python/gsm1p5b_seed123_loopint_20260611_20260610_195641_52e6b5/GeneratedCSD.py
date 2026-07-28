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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. At the very end of your response, write the final arithmetic expression (using only numbers and operators +,-,*,/,(,)) inside << >> delimiters. Use << >> only once, for the final answer.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_forceThreshold_: int
        if (maxSteps) > (100):
            d_2_forceThreshold_ = (maxSteps) - (100)
        elif True:
            d_2_forceThreshold_ = 0
        d_3_closeReserve_: int
        d_3_closeReserve_ = 2
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_1_steps_) >= (d_2_forceThreshold_)) and (((d_1_steps_) + (d_3_closeReserve_)) < (maxSteps)):
                            d_4_og_: _dafny.Seq
                            d_5_oi_: bool
                            d_6_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_og_ = out0_
                            d_5_oi_ = out1_
                            d_6_oc_ = out2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            generated = d_4_og_
                            insideConstrainedOut = d_5_oi_
                            currentConstrainedOut = d_6_oc_
                        elif True:
                            d_7_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_7_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                                if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_8_og_: _dafny.Seq
                                    d_9_oi_: bool
                                    d_10_oc_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_8_og_ = out4_
                                    d_9_oi_ = out5_
                                    d_10_oc_ = out6_
                                    generated = d_8_og_
                                    insideConstrainedOut = d_9_oi_
                                    currentConstrainedOut = d_10_oc_
                    elif True:
                        if ((d_1_steps_) + (d_3_closeReserve_)) >= (maxSteps):
                            d_11_cg_: _dafny.Seq
                            d_12_ci_: bool
                            d_13_cc_: _dafny.Seq
                            d_14_closed_: bool
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out10_: bool
                            out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_11_cg_ = out7_
                            d_12_ci_ = out8_
                            d_13_cc_ = out9_
                            d_14_closed_ = out10_
                            d_1_steps_ = (d_1_steps_) + (1)
                            generated = d_11_cg_
                            insideConstrainedOut = d_12_ci_
                            currentConstrainedOut = d_13_cc_
                            if d_14_closed_:
                                raise _dafny.Break("0")
                            if (d_1_steps_) >= (maxSteps):
                                raise _dafny.Break("0")
                        elif True:
                            d_15_cg_: _dafny.Seq
                            d_16_ci_: bool
                            d_17_cc_: _dafny.Seq
                            d_18_closed_: bool
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out14_: bool
                            out11_, out12_, out13_, out14_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_15_cg_ = out11_
                            d_16_ci_ = out12_
                            d_17_cc_ = out13_
                            d_18_closed_ = out14_
                            d_1_steps_ = (d_1_steps_) + (1)
                            generated = d_15_cg_
                            insideConstrainedOut = d_16_ci_
                            currentConstrainedOut = d_17_cc_
                            if d_18_closed_:
                                pass
                            elif True:
                                if (d_1_steps_) < (maxSteps):
                                    d_19_constrainedPrompt_: _dafny.Seq
                                    d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_20_next_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_20_next_ = out15_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_20_next_) == (eosToken):
                                        d_21_rg_: _dafny.Seq
                                        d_22_rc_: _dafny.Seq
                                        out16_: _dafny.Seq
                                        out17_: _dafny.Seq
                                        out16_, out17_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                        d_21_rg_ = out16_
                                        d_22_rc_ = out17_
                                        generated = d_21_rg_
                                        currentConstrainedOut = d_22_rc_
                                        if (parser).IsCompletePrefix(currentConstrainedOut):
                                            d_23_fg_: _dafny.Seq
                                            d_24_fi_: bool
                                            d_25_fc_: _dafny.Seq
                                            d_26_fclosed_: bool
                                            out18_: _dafny.Seq
                                            out19_: bool
                                            out20_: _dafny.Seq
                                            out21_: bool
                                            out18_, out19_, out20_, out21_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                            d_23_fg_ = out18_
                                            d_24_fi_ = out19_
                                            d_25_fc_ = out20_
                                            d_26_fclosed_ = out21_
                                            if (d_1_steps_) < (maxSteps):
                                                d_1_steps_ = (d_1_steps_) + (1)
                                                generated = d_23_fg_
                                                insideConstrainedOut = d_24_fi_
                                                currentConstrainedOut = d_25_fc_
                                        elif True:
                                            insideConstrainedOut = False
                                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_27_ag_: _dafny.Seq
                                        d_28_ai_: bool
                                        d_29_ac_: _dafny.Seq
                                        out22_: _dafny.Seq
                                        out23_: bool
                                        out24_: _dafny.Seq
                                        out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                                        d_27_ag_ = out22_
                                        d_28_ai_ = out23_
                                        d_29_ac_ = out24_
                                        generated = d_27_ag_
                                        insideConstrainedOut = d_28_ai_
                                        currentConstrainedOut = d_29_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

