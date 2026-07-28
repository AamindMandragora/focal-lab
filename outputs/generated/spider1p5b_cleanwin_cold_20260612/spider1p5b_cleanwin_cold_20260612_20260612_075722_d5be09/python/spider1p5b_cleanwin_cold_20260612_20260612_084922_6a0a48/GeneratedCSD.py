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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single SQL query. Output format must be exactly: SQL: <<SELECT ...>> with no other text. Write concise correct SQL using the schema provided.")))
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_2_chunkMax_: int
            d_2_chunkMax_ = 10
            if (d_2_chunkMax_) > ((maxSteps) - (d_1_steps_)):
                d_2_chunkMax_ = (maxSteps) - (d_1_steps_)
            if (d_2_chunkMax_) > (0):
                d_3_genOut_: _dafny.Seq
                d_4_stoppedOnOpen_: bool
                d_5_stoppedOnEos_: bool
                d_6_stepsUsed_: int
                out0_: _dafny.Seq
                out1_: bool
                out2_: bool
                out3_: int
                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkMax_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                d_3_genOut_ = out0_
                d_4_stoppedOnOpen_ = out1_
                d_5_stoppedOnEos_ = out2_
                d_6_stepsUsed_ = out3_
                d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed_)
                generated = d_3_genOut_
                if d_5_stoppedOnEos_:
                    cost = d_1_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                if d_4_stoppedOnOpen_:
                    d_7_g2_: _dafny.Seq
                    d_8_i2_: bool
                    d_9_c2_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                    d_7_g2_ = out4_
                    d_8_i2_ = out5_
                    d_9_c2_ = out6_
                    generated = d_7_g2_
                    insideConstrainedOut = d_8_i2_
                    currentConstrainedOut = d_9_c2_
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_10_g2_: _dafny.Seq
            d_11_i2_: bool
            d_12_c2_: _dafny.Seq
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_10_g2_ = out7_
            d_11_i2_ = out8_
            d_12_c2_ = out9_
            generated = d_10_g2_
            insideConstrainedOut = d_11_i2_
            currentConstrainedOut = d_12_c2_
            d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    d_13_cg_: _dafny.Seq
                    d_14_ci_: bool
                    d_15_cc_: _dafny.Seq
                    d_16_closed_: bool
                    out10_: _dafny.Seq
                    out11_: bool
                    out12_: _dafny.Seq
                    out13_: bool
                    out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_13_cg_ = out10_
                    d_14_ci_ = out11_
                    d_15_cc_ = out12_
                    d_16_closed_ = out13_
                    if d_16_closed_:
                        generated = d_13_cg_
                        insideConstrainedOut = d_14_ci_
                        currentConstrainedOut = d_15_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    if (d_1_steps_) >= (maxSteps):
                        raise _dafny.Break("0")
                    d_17_constrainedPrompt_: _dafny.Seq
                    d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_18_next_: _dafny.Seq
                    out14_: _dafny.Seq
                    out14_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                    d_18_next_ = out14_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_18_next_) == (eosToken):
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_19_cg2_: _dafny.Seq
                            d_20_ci2_: bool
                            d_21_cc2_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_19_cg2_ = out15_
                            d_20_ci2_ = out16_
                            d_21_cc2_ = out17_
                            generated = d_19_cg2_
                            insideConstrainedOut = d_20_ci2_
                            currentConstrainedOut = d_21_cc2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_22_ag_: _dafny.Seq
                        d_23_ai_: bool
                        d_24_ac_: _dafny.Seq
                        out18_: _dafny.Seq
                        out19_: bool
                        out20_: _dafny.Seq
                        out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                        d_22_ag_ = out18_
                        d_23_ai_ = out19_
                        d_24_ac_ = out20_
                        generated = d_22_ag_
                        insideConstrainedOut = d_23_ai_
                        currentConstrainedOut = d_24_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_25_rg_: _dafny.Seq
            d_26_rc_: _dafny.Seq
            out21_: _dafny.Seq
            out22_: _dafny.Seq
            out21_, out22_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_25_rg_ = out21_
            d_26_rc_ = out22_
            if (parser).IsCompletePrefix(d_26_rc_):
                generated = d_25_rg_
                currentConstrainedOut = d_26_rc_
                insideConstrainedOut = True
                if (d_1_steps_) < (maxSteps):
                    d_27_cg3_: _dafny.Seq
                    d_28_ci3_: bool
                    d_29_cc3_: _dafny.Seq
                    out23_: _dafny.Seq
                    out24_: bool
                    out25_: _dafny.Seq
                    out23_, out24_, out25_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_27_cg3_ = out23_
                    d_28_ci3_ = out24_
                    d_29_cc3_ = out25_
                    generated = d_27_cg3_
                    insideConstrainedOut = d_28_ci3_
                    currentConstrainedOut = d_29_cc3_
                    d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

