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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Task: generate one valid SMILES for a chain_extenders molecule. Chain extenders are small bifunctional molecules used in polyurethane synthesis. Examples include 1,4-butanediol (OCCCCO), ethylene glycol (OCCO), 1,3-propanediol (OCCCO), hexamethylene diamine (NCCCCCCN), ethylenediamine (NCCN), 1,3-diaminopropane (NCCCN), diethanolamine (OCCNCCО). Output ONLY a SMILES string that is different from these examples.")))
        if (not(insideConstrainedOut)) and (((d_1_steps_) + (2)) <= (maxSteps)):
            d_2_og_: _dafny.Seq
            d_3_oi_: bool
            d_4_oc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_2_og_ = out0_
            d_3_oi_ = out1_
            d_4_oc_ = out2_
            generated = d_2_og_
            insideConstrainedOut = d_3_oi_
            currentConstrainedOut = d_4_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (6)):
                            d_5_cg_: _dafny.Seq
                            d_6_ci_: bool
                            d_7_cc_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_5_cg_ = out3_
                            d_6_ci_ = out4_
                            d_7_cc_ = out5_
                            generated = d_5_cg_
                            insideConstrainedOut = d_6_ci_
                            currentConstrainedOut = d_7_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif (parser).IsCompletePrefix(currentConstrainedOut):
                            d_8_constrainedPrompt_: _dafny.Seq
                            d_8_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_9_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('3e0'), eosToken)
                            d_9_next_ = out6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_10_cg_: _dafny.Seq
                                    d_11_ci_: bool
                                    d_12_cc_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_10_cg_ = out7_
                                    d_11_ci_ = out8_
                                    d_12_cc_ = out9_
                                    generated = d_10_cg_
                                    insideConstrainedOut = d_11_ci_
                                    currentConstrainedOut = d_12_cc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                d_13_isComplete_: bool
                                d_13_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if not(d_13_isComplete_):
                                    d_14_ag_: _dafny.Seq
                                    d_15_ai_: bool
                                    d_16_ac_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                                    d_14_ag_ = out10_
                                    d_15_ai_ = out11_
                                    d_16_ac_ = out12_
                                    generated = d_14_ag_
                                    insideConstrainedOut = d_15_ai_
                                    currentConstrainedOut = d_16_ac_
                                elif True:
                                    if (d_1_steps_) < (maxSteps):
                                        d_17_cg_: _dafny.Seq
                                        d_18_ci_: bool
                                        d_19_cc_: _dafny.Seq
                                        out13_: _dafny.Seq
                                        out14_: bool
                                        out15_: _dafny.Seq
                                        out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_17_cg_ = out13_
                                        d_18_ci_ = out14_
                                        d_19_cc_ = out15_
                                        generated = d_17_cg_
                                        insideConstrainedOut = d_18_ci_
                                        currentConstrainedOut = d_19_cc_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    raise _dafny.Break("0")
                        elif True:
                            d_20_constrainedPrompt_: _dafny.Seq
                            d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_21_next_: _dafny.Seq
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), 50, eosToken)
                            d_21_next_ = out16_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_21_next_) == (eosToken):
                                d_22_rg_: _dafny.Seq
                                d_23_rc_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: _dafny.Seq
                                out17_, out18_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_22_rg_ = out17_
                                d_23_rc_ = out18_
                                generated = d_22_rg_
                                currentConstrainedOut = d_23_rc_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_24_cg_: _dafny.Seq
                                    d_25_ci_: bool
                                    d_26_cc_: _dafny.Seq
                                    out19_: _dafny.Seq
                                    out20_: bool
                                    out21_: _dafny.Seq
                                    out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_24_cg_ = out19_
                                    d_25_ci_ = out20_
                                    d_26_cc_ = out21_
                                    generated = d_24_cg_
                                    insideConstrainedOut = d_25_ci_
                                    currentConstrainedOut = d_26_cc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                d_27_isComplete_: bool
                                d_27_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if not(d_27_isComplete_):
                                    d_28_ag_: _dafny.Seq
                                    d_29_ai_: bool
                                    d_30_ac_: _dafny.Seq
                                    out22_: _dafny.Seq
                                    out23_: bool
                                    out24_: _dafny.Seq
                                    out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                    d_28_ag_ = out22_
                                    d_29_ai_ = out23_
                                    d_30_ac_ = out24_
                                    generated = d_28_ag_
                                    insideConstrainedOut = d_29_ai_
                                    currentConstrainedOut = d_30_ac_
                    elif True:
                        d_31_next_: _dafny.Seq
                        out25_: _dafny.Seq
                        out25_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_31_next_ = out25_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_31_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_31_next_]))
                    pass
            pass
        if insideConstrainedOut:
            d_32_rg_: _dafny.Seq
            d_33_rc_: _dafny.Seq
            out26_: _dafny.Seq
            out27_: _dafny.Seq
            out26_, out27_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_32_rg_ = out26_
            d_33_rc_ = out27_
            generated = d_32_rg_
            currentConstrainedOut = d_33_rc_
            if ((d_1_steps_) < (maxSteps)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                d_34_cg_: _dafny.Seq
                d_35_ci_: bool
                d_36_cc_: _dafny.Seq
                out28_: _dafny.Seq
                out29_: bool
                out30_: _dafny.Seq
                out28_, out29_, out30_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_34_cg_ = out28_
                d_35_ci_ = out29_
                d_36_cc_ = out30_
                generated = d_34_cg_
                insideConstrainedOut = d_35_ci_
                currentConstrainedOut = d_36_cc_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

