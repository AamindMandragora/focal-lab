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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one valid SQL query in the format: SQL: <<YOUR SQL QUERY>>. Generate a single valid SQL query inside the constrained span. Use only schema identifiers from the context.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_2_chunkBudget_: int
            d_2_chunkBudget_ = 40
            if (d_2_chunkBudget_) > ((maxSteps) - (d_1_steps_)):
                d_2_chunkBudget_ = (maxSteps) - (d_1_steps_)
            d_3_chunkGen_: _dafny.Seq
            d_4_stoppedOnOpen_: bool
            d_5_stoppedOnEos_: bool
            d_6_chunkSteps_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_3_chunkGen_ = out0_
            d_4_stoppedOnOpen_ = out1_
            d_5_stoppedOnEos_ = out2_
            d_6_chunkSteps_ = out3_
            d_1_steps_ = (d_1_steps_) + (d_6_chunkSteps_)
            generated = d_3_chunkGen_
            if d_5_stoppedOnEos_:
                cost = d_1_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            if d_4_stoppedOnOpen_:
                d_7_eg_: _dafny.Seq
                d_8_ei_: bool
                d_9_ec_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_7_eg_ = out4_
                d_8_ei_ = out5_
                d_9_ec_ = out6_
                generated = d_7_eg_
                insideConstrainedOut = d_8_ei_
                currentConstrainedOut = d_9_ec_
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_10_chunkBudget2_: int
            d_10_chunkBudget2_ = 20
            if (d_10_chunkBudget2_) > ((maxSteps) - (d_1_steps_)):
                d_10_chunkBudget2_ = (maxSteps) - (d_1_steps_)
            d_11_chunkGen2_: _dafny.Seq
            d_12_stoppedOnOpen2_: bool
            d_13_stoppedOnEos2_: bool
            d_14_chunkSteps2_: int
            out7_: _dafny.Seq
            out8_: bool
            out9_: bool
            out10_: int
            out7_, out8_, out9_, out10_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_10_chunkBudget2_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_11_chunkGen2_ = out7_
            d_12_stoppedOnOpen2_ = out8_
            d_13_stoppedOnEos2_ = out9_
            d_14_chunkSteps2_ = out10_
            d_1_steps_ = (d_1_steps_) + (d_14_chunkSteps2_)
            generated = d_11_chunkGen2_
            if d_13_stoppedOnEos2_:
                cost = d_1_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            if d_12_stoppedOnOpen2_:
                d_15_eg2_: _dafny.Seq
                d_16_ei2_: bool
                d_17_ec2_: _dafny.Seq
                out11_: _dafny.Seq
                out12_: bool
                out13_: _dafny.Seq
                out11_, out12_, out13_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_15_eg2_ = out11_
                d_16_ei2_ = out12_
                d_17_ec2_ = out13_
                generated = d_15_eg2_
                insideConstrainedOut = d_16_ei2_
                currentConstrainedOut = d_17_ec2_
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_18_og_: _dafny.Seq
            d_19_oi_: bool
            d_20_oc_: _dafny.Seq
            out14_: _dafny.Seq
            out15_: bool
            out16_: _dafny.Seq
            out14_, out15_, out16_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_18_og_ = out14_
            d_19_oi_ = out15_
            d_20_oc_ = out16_
            generated = d_18_og_
            insideConstrainedOut = d_19_oi_
            currentConstrainedOut = d_20_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if ((d_1_steps_) + (5)) >= (maxSteps):
                        raise _dafny.Break("0")
                    d_21_cg_: _dafny.Seq
                    d_22_ci_: bool
                    d_23_cc_: _dafny.Seq
                    d_24_closed_: bool
                    out17_: _dafny.Seq
                    out18_: bool
                    out19_: _dafny.Seq
                    out20_: bool
                    out17_, out18_, out19_, out20_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_21_cg_ = out17_
                    d_22_ci_ = out18_
                    d_23_cc_ = out19_
                    d_24_closed_ = out20_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if d_24_closed_:
                        generated = d_21_cg_
                        insideConstrainedOut = d_22_ci_
                        currentConstrainedOut = d_23_cc_
                        raise _dafny.Break("0")
                    elif True:
                        d_25_constrainedPrompt_: _dafny.Seq
                        d_25_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_26_next_: _dafny.Seq
                        out21_: _dafny.Seq
                        out21_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_25_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('3e0'), eosToken)
                        d_26_next_ = out21_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_26_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_27_ag_: _dafny.Seq
                            d_28_ai_: bool
                            d_29_ac_: _dafny.Seq
                            out22_: _dafny.Seq
                            out23_: bool
                            out24_: _dafny.Seq
                            out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next_)
                            d_27_ag_ = out22_
                            d_28_ai_ = out23_
                            d_29_ac_ = out24_
                            generated = d_27_ag_
                            insideConstrainedOut = d_28_ai_
                            currentConstrainedOut = d_29_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_30_closeBudget_: int
            d_30_closeBudget_ = (maxSteps) - (d_1_steps_)
            if (d_30_closeBudget_) > (80):
                d_30_closeBudget_ = 80
            d_31_fg_: _dafny.Seq
            d_32_fi_: bool
            d_33_fc_: _dafny.Seq
            out25_: _dafny.Seq
            out26_: bool
            out27_: _dafny.Seq
            out25_, out26_, out27_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_30_closeBudget_)
            d_31_fg_ = out25_
            d_32_fi_ = out26_
            d_33_fc_ = out27_
            generated = d_31_fg_
            insideConstrainedOut = d_32_fi_
            currentConstrainedOut = d_33_fc_
            d_1_steps_ = (d_1_steps_) + (d_30_closeBudget_)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

