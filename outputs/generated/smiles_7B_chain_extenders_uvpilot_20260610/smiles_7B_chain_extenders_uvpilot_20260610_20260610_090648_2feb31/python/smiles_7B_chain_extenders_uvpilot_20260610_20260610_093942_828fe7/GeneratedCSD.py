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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a novel valid SMILES string for a chain_extenders molecule. Chain extenders are small bifunctional molecules with two reactive end groups such as -OH or -NH2. Generate a NEW molecule with at least 4 heavy atoms. Output only the SMILES string.")))
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
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
                        d_5_isComplete_: bool
                        d_5_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_5_isComplete_:
                            d_6_cg_: _dafny.Seq
                            d_7_ci_: bool
                            d_8_cc_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_6_cg_ = out3_
                            d_7_ci_ = out4_
                            d_8_cc_ = out5_
                            generated = d_6_cg_
                            insideConstrainedOut = d_7_ci_
                            currentConstrainedOut = d_8_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_9_constrainedPrompt_: _dafny.Seq
                            d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_10_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('3e0'), eosToken)
                            d_10_next_ = out6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_10_next_) == (eosToken):
                                d_11_rg_: _dafny.Seq
                                d_12_rc_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: _dafny.Seq
                                out7_, out8_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_11_rg_ = out7_
                                d_12_rc_ = out8_
                                generated = d_11_rg_
                                currentConstrainedOut = d_12_rc_
                                d_13_isCompleteAfterRollback_: bool
                                d_13_isCompleteAfterRollback_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if (d_13_isCompleteAfterRollback_) and ((d_1_steps_) < (maxSteps)):
                                    d_14_cg_: _dafny.Seq
                                    d_15_ci_: bool
                                    d_16_cc_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_14_cg_ = out9_
                                    d_15_ci_ = out10_
                                    d_16_cc_ = out11_
                                    generated = d_14_cg_
                                    insideConstrainedOut = d_15_ci_
                                    currentConstrainedOut = d_16_cc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                            elif True:
                                d_17_notComplete_: bool
                                d_17_notComplete_ = not((parser).IsCompletePrefix(currentConstrainedOut))
                                if d_17_notComplete_:
                                    d_18_ag_: _dafny.Seq
                                    d_19_ai_: bool
                                    d_20_ac_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                                    d_18_ag_ = out12_
                                    d_19_ai_ = out13_
                                    d_20_ac_ = out14_
                                    generated = d_18_ag_
                                    insideConstrainedOut = d_19_ai_
                                    currentConstrainedOut = d_20_ac_
                                elif True:
                                    d_21_cg_: _dafny.Seq
                                    d_22_ci_: bool
                                    d_23_cc_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_21_cg_ = out15_
                                    d_22_ci_ = out16_
                                    d_23_cc_ = out17_
                                    generated = d_21_cg_
                                    insideConstrainedOut = d_22_ci_
                                    currentConstrainedOut = d_23_cc_
                                    raise _dafny.Break("0")
                    elif True:
                        d_24_next_: _dafny.Seq
                        out18_: _dafny.Seq
                        out18_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_24_next_ = out18_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_24_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_24_next_]))
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_25_rg_: _dafny.Seq
            d_26_rc_: _dafny.Seq
            out19_: _dafny.Seq
            out20_: _dafny.Seq
            out19_, out20_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_25_rg_ = out19_
            d_26_rc_ = out20_
            generated = d_25_rg_
            currentConstrainedOut = d_26_rc_
            d_27_isFinalComplete_: bool
            d_27_isFinalComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if (d_27_isFinalComplete_) and ((d_1_steps_) < (maxSteps)):
                d_28_cg_: _dafny.Seq
                d_29_ci_: bool
                d_30_cc_: _dafny.Seq
                out21_: _dafny.Seq
                out22_: bool
                out23_: _dafny.Seq
                out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_28_cg_ = out21_
                d_29_ci_ = out22_
                d_30_cc_ = out23_
                generated = d_28_cg_
                insideConstrainedOut = d_29_ci_
                currentConstrainedOut = d_30_cc_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

