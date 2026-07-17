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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single concise SQL query answering the question. Use only table and column names from the schema. Format: SQL: <<SELECT ...>>. Keep the query simple and correct.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_2_chunkBudget_: int
            d_2_chunkBudget_ = 6
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
            d_10_og_: _dafny.Seq
            d_11_oi_: bool
            d_12_oc_: _dafny.Seq
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_10_og_ = out7_
            d_11_oi_ = out8_
            d_12_oc_ = out9_
            generated = d_10_og_
            insideConstrainedOut = d_11_oi_
            currentConstrainedOut = d_12_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        d_13_spanLength_: int
        d_13_spanLength_ = 0
        d_14_spanCap_: int
        d_14_spanCap_ = 65
        with _dafny.label("0"):
            while (((d_1_steps_) < (maxSteps)) and (insideConstrainedOut)) and ((d_13_spanLength_) < (d_14_spanCap_)):
                with _dafny.c_label("0"):
                    if ((d_1_steps_) + (20)) >= (maxSteps):
                        raise _dafny.Break("0")
                    d_15_cg_: _dafny.Seq
                    d_16_ci_: bool
                    d_17_cc_: _dafny.Seq
                    d_18_closed_: bool
                    out10_: _dafny.Seq
                    out11_: bool
                    out12_: _dafny.Seq
                    out13_: bool
                    out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_15_cg_ = out10_
                    d_16_ci_ = out11_
                    d_17_cc_ = out12_
                    d_18_closed_ = out13_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if d_18_closed_:
                        generated = d_15_cg_
                        insideConstrainedOut = d_16_ci_
                        currentConstrainedOut = d_17_cc_
                        raise _dafny.Break("0")
                    elif True:
                        d_19_constrainedPrompt_: _dafny.Seq
                        d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_20_validCount_: int
                        out14_: int
                        out14_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_20_validCount_ = out14_
                        d_21_next_: _dafny.Seq
                        d_21_next_ = eosToken
                        if (d_20_validCount_) <= (12):
                            out15_: _dafny.Seq
                            out15_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_21_next_ = out15_
                        elif True:
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_21_next_ = out16_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_1_steps_) > (maxSteps):
                            d_1_steps_ = maxSteps
                        if (d_21_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_22_ag_: _dafny.Seq
                            d_23_ai_: bool
                            d_24_ac_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: bool
                            out19_: _dafny.Seq
                            out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                            d_22_ag_ = out17_
                            d_23_ai_ = out18_
                            d_24_ac_ = out19_
                            generated = d_22_ag_
                            insideConstrainedOut = d_23_ai_
                            currentConstrainedOut = d_24_ac_
                            d_13_spanLength_ = (d_13_spanLength_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_25_closeBudget_: int
            d_25_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_26_fg_: _dafny.Seq
            d_27_fi_: bool
            d_28_fc_: _dafny.Seq
            out20_: _dafny.Seq
            out21_: bool
            out22_: _dafny.Seq
            out20_, out21_, out22_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_25_closeBudget_)
            d_26_fg_ = out20_
            d_27_fi_ = out21_
            d_28_fc_ = out22_
            generated = d_26_fg_
            insideConstrainedOut = d_27_fi_
            currentConstrainedOut = d_28_fc_
            d_1_steps_ = (d_1_steps_) + (d_25_closeBudget_)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

