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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer with exactly: SQL: <<your_sql_query>>. Output only that line. No explanation. No markdown. The SQL query goes between << and >>. Example: SQL: <<SELECT column FROM table WHERE condition>>"))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_closeReserve_: int
        d_3_closeReserve_ = 2
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out0_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if ((d_2_steps_) + (d_3_closeReserve_)) >= (maxSteps):
                            d_5_rg_: _dafny.Seq
                            d_6_rc_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: _dafny.Seq
                            out1_, out2_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_5_rg_ = out1_
                            d_6_rc_ = out2_
                            generated = d_5_rg_
                            currentConstrainedOut = d_6_rc_
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_7_cg_: _dafny.Seq
                                d_8_ci_: bool
                                d_9_cc_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_7_cg_ = out3_
                                d_8_ci_ = out4_
                                d_9_cc_ = out5_
                                d_2_steps_ = (d_2_steps_) + (1)
                                generated = d_7_cg_
                                insideConstrainedOut = d_8_ci_
                                currentConstrainedOut = d_9_cc_
                            elif True:
                                d_2_steps_ = (d_2_steps_) + (1)
                            raise _dafny.Break("0")
                        d_10_cg_: _dafny.Seq
                        d_11_ci_: bool
                        d_12_cc_: _dafny.Seq
                        d_13_closed_: bool
                        out6_: _dafny.Seq
                        out7_: bool
                        out8_: _dafny.Seq
                        out9_: bool
                        out6_, out7_, out8_, out9_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_10_cg_ = out6_
                        d_11_ci_ = out7_
                        d_12_cc_ = out8_
                        d_13_closed_ = out9_
                        if d_13_closed_:
                            d_2_steps_ = (d_2_steps_) + (1)
                            generated = d_10_cg_
                            insideConstrainedOut = d_11_ci_
                            currentConstrainedOut = d_12_cc_
                            raise _dafny.Break("0")
                        elif True:
                            d_14_constrainedPrompt_: _dafny.Seq
                            d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_15_next_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_15_next_ = out10_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_15_next_) == (eosToken):
                                d_16_rg2_: _dafny.Seq
                                d_17_rc2_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: _dafny.Seq
                                out11_, out12_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_16_rg2_ = out11_
                                d_17_rc2_ = out12_
                                generated = d_16_rg2_
                                currentConstrainedOut = d_17_rc2_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                    d_18_cg2_: _dafny.Seq
                                    d_19_ci2_: bool
                                    d_20_cc2_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out15_: _dafny.Seq
                                    out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_18_cg2_ = out13_
                                    d_19_ci2_ = out14_
                                    d_20_cc2_ = out15_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    generated = d_18_cg2_
                                    insideConstrainedOut = d_19_ci2_
                                    currentConstrainedOut = d_20_cc2_
                                raise _dafny.Break("0")
                            elif True:
                                d_21_ag_: _dafny.Seq
                                d_22_ai_: bool
                                d_23_ac_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                d_21_ag_ = out16_
                                d_22_ai_ = out17_
                                d_23_ac_ = out18_
                                generated = d_21_ag_
                                insideConstrainedOut = d_22_ai_
                                currentConstrainedOut = d_23_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_24_rg3_: _dafny.Seq
            d_25_rc3_: _dafny.Seq
            out19_: _dafny.Seq
            out20_: _dafny.Seq
            out19_, out20_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_24_rg3_ = out19_
            d_25_rc3_ = out20_
            generated = d_24_rg3_
            currentConstrainedOut = d_25_rc3_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                d_26_cg3_: _dafny.Seq
                d_27_ci3_: bool
                d_28_cc3_: _dafny.Seq
                out21_: _dafny.Seq
                out22_: bool
                out23_: _dafny.Seq
                out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_26_cg3_ = out21_
                d_27_ci3_ = out22_
                d_28_cc3_ = out23_
                d_2_steps_ = (d_2_steps_) + (1)
                generated = d_26_cg3_
                insideConstrainedOut = d_27_ci3_
                currentConstrainedOut = d_28_cc3_
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

