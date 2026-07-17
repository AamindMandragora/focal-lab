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
        d_1_steps_: int
        d_1_steps_ = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write a single correct SQL query for the given question using the schema. Output only the SQL query inside the constrained span.")))
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_2_g2_: _dafny.Seq
            d_3_i2_: bool
            d_4_c2_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_2_g2_ = out0_
            d_3_i2_ = out1_
            d_4_c2_ = out2_
            generated = d_2_g2_
            insideConstrainedOut = d_3_i2_
            currentConstrainedOut = d_4_c2_
            d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    if ((len(currentConstrainedOut)) >= (45)) and (((d_1_steps_) + (2)) <= (maxSteps)):
                        d_5_rg_: _dafny.Seq
                        d_6_rc_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: _dafny.Seq
                        out3_, out4_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_5_rg_ = out3_
                        d_6_rc_ = out4_
                        if (parser).IsCompletePrefix(d_6_rc_):
                            generated = d_5_rg_
                            currentConstrainedOut = d_6_rc_
                            insideConstrainedOut = True
                            d_7_cg3_: _dafny.Seq
                            d_8_ci3_: bool
                            d_9_cc3_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_7_cg3_ = out5_
                            d_8_ci3_ = out6_
                            d_9_cc3_ = out7_
                            generated = d_7_cg3_
                            insideConstrainedOut = d_8_ci3_
                            currentConstrainedOut = d_9_cc3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                    d_10_cg_: _dafny.Seq
                    d_11_ci_: bool
                    d_12_cc_: _dafny.Seq
                    d_13_closed_: bool
                    out8_: _dafny.Seq
                    out9_: bool
                    out10_: _dafny.Seq
                    out11_: bool
                    out8_, out9_, out10_, out11_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_10_cg_ = out8_
                    d_11_ci_ = out9_
                    d_12_cc_ = out10_
                    d_13_closed_ = out11_
                    if d_13_closed_:
                        generated = d_10_cg_
                        insideConstrainedOut = d_11_ci_
                        currentConstrainedOut = d_12_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    if (d_1_steps_) >= (maxSteps):
                        raise _dafny.Break("0")
                    d_14_constrainedPrompt_: _dafny.Seq
                    d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_15_next_: _dafny.Seq
                    out12_: _dafny.Seq
                    out12_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), eosToken)
                    d_15_next_ = out12_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_15_next_) == (eosToken):
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_16_cg2_: _dafny.Seq
                            d_17_ci2_: bool
                            d_18_cc2_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_16_cg2_ = out13_
                            d_17_ci2_ = out14_
                            d_18_cc2_ = out15_
                            generated = d_16_cg2_
                            insideConstrainedOut = d_17_ci2_
                            currentConstrainedOut = d_18_cc2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_19_ag_: _dafny.Seq
                        d_20_ai_: bool
                        d_21_ac_: _dafny.Seq
                        out16_: _dafny.Seq
                        out17_: bool
                        out18_: _dafny.Seq
                        out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                        d_19_ag_ = out16_
                        d_20_ai_ = out17_
                        d_21_ac_ = out18_
                        generated = d_19_ag_
                        insideConstrainedOut = d_20_ai_
                        currentConstrainedOut = d_21_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_22_rg_: _dafny.Seq
            d_23_rc_: _dafny.Seq
            out19_: _dafny.Seq
            out20_: _dafny.Seq
            out19_, out20_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_22_rg_ = out19_
            d_23_rc_ = out20_
            if (parser).IsCompletePrefix(d_23_rc_):
                generated = d_22_rg_
                currentConstrainedOut = d_23_rc_
                insideConstrainedOut = True
                if (d_1_steps_) < (maxSteps):
                    d_24_cg3_: _dafny.Seq
                    d_25_ci3_: bool
                    d_26_cc3_: _dafny.Seq
                    out21_: _dafny.Seq
                    out22_: bool
                    out23_: _dafny.Seq
                    out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_24_cg3_ = out21_
                    d_25_ci3_ = out22_
                    d_26_cc3_ = out23_
                    generated = d_24_cg3_
                    insideConstrainedOut = d_25_ci3_
                    currentConstrainedOut = d_26_cc3_
                    d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

