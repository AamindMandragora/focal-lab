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
        out0_: _dafny.Seq
        out1_: bool
        out2_: _dafny.Seq
        out3_: int
        out0_, out1_, out2_, out3_ = default__.AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken)
        generated = out0_
        insideConstrainedOut = out1_
        currentConstrainedOut = out2_
        cost = out3_
        if ((maxSteps) > (0)) and ((cost) <= (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

    @staticmethod
    def AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Task: output exactly one SMILES string for a novel chain_extender molecule. Chain extenders are bifunctional small molecules (2 OH groups, 2 NH2 groups, or 1 OH + 1 NH2). Required: the SMILES must have at least 4 heavy atoms. Use structures like: OCCCCO (1,4-butanediol), OCCO (ethylene glycol), NCCCCN (putrescine), NCCO (ethanolamine), OCC(CO)CO, OCCOCCCO. Prefer 4-8 carbon chain lengths. Do NOT output single atoms like C, N, O. Output ONLY the SMILES, nothing else."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_minSpanLength_: int
        d_3_minSpanLength_ = 4
        d_4_preambleMax_: int
        d_4_preambleMax_ = 3
        d_5_preambleSteps_: int
        d_5_preambleSteps_ = 0
        with _dafny.label("0"):
            while (((d_2_steps_) < (maxSteps)) and ((d_5_preambleSteps_) < (d_4_preambleMax_))) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_6_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_6_next_ = out0_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_5_preambleSteps_ = (d_5_preambleSteps_) + (1)
                    if (d_6_next_) == (eosToken):
                        raise _dafny.Break("0")
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                    if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
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
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_10_og_: _dafny.Seq
            d_11_oi_: bool
            d_12_oc_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_10_og_ = out4_
            d_11_oi_ = out5_
            d_12_oc_ = out6_
            generated = d_10_og_
            insideConstrainedOut = d_11_oi_
            currentConstrainedOut = d_12_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        with _dafny.label("1"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("1"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("1")
                    d_13_spanLen_: int
                    d_13_spanLen_ = len(currentConstrainedOut)
                    d_14_isComplete_: bool
                    d_14_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                    d_15_validCount_: int
                    out7_: int
                    out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                    d_15_validCount_ = out7_
                    if (d_14_isComplete_) and (((d_13_spanLen_) >= (d_3_minSpanLength_)) or ((d_15_validCount_) == (0))):
                        d_16_cg_: _dafny.Seq
                        d_17_ci_: bool
                        d_18_cc_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_16_cg_ = out8_
                        d_17_ci_ = out9_
                        d_18_cc_ = out10_
                        generated = d_16_cg_
                        insideConstrainedOut = d_17_ci_
                        currentConstrainedOut = d_18_cc_
                        d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("1")
                    d_19_constrainedPrompt_: _dafny.Seq
                    d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_20_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                    if (d_13_spanLen_) < (d_3_minSpanLength_):
                        out11_: _dafny.Seq
                        out11_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('4e0'), eosToken)
                        d_20_next_ = out11_
                    elif (d_15_validCount_) <= (6):
                        out12_: _dafny.Seq
                        out12_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), eosToken)
                        d_20_next_ = out12_
                    elif (d_15_validCount_) <= (20):
                        out13_: _dafny.Seq
                        out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 20, eosToken)
                        d_20_next_ = out13_
                    elif True:
                        out14_: _dafny.Seq
                        out14_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                        d_20_next_ = out14_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_20_next_) == (eosToken):
                        d_21_isCompleteNow_: bool
                        d_21_isCompleteNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if (d_21_isCompleteNow_) and ((d_2_steps_) < (maxSteps)):
                            d_22_cg_: _dafny.Seq
                            d_23_ci_: bool
                            d_24_cc_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_22_cg_ = out15_
                            d_23_ci_ = out16_
                            d_24_cc_ = out17_
                            generated = d_22_cg_
                            insideConstrainedOut = d_23_ci_
                            currentConstrainedOut = d_24_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("1")
                    d_25_isCompleteBeforeAppend_: bool
                    d_25_isCompleteBeforeAppend_ = (parser).IsCompletePrefix(currentConstrainedOut)
                    if not(d_25_isCompleteBeforeAppend_):
                        d_26_ag_: _dafny.Seq
                        d_27_ai_: bool
                        d_28_ac_: _dafny.Seq
                        out18_: _dafny.Seq
                        out19_: bool
                        out20_: _dafny.Seq
                        out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                        d_26_ag_ = out18_
                        d_27_ai_ = out19_
                        d_28_ac_ = out20_
                        generated = d_26_ag_
                        insideConstrainedOut = d_27_ai_
                        currentConstrainedOut = d_28_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_29_isFinalComplete_: bool
            d_29_isFinalComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if d_29_isFinalComplete_:
                d_30_cg_: _dafny.Seq
                d_31_ci_: bool
                d_32_cc_: _dafny.Seq
                out21_: _dafny.Seq
                out22_: bool
                out23_: _dafny.Seq
                out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_30_cg_ = out21_
                d_31_ci_ = out22_
                d_32_cc_ = out23_
                generated = d_30_cg_
                insideConstrainedOut = d_31_ci_
                currentConstrainedOut = d_32_cc_
                d_2_steps_ = (d_2_steps_) + (1)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

