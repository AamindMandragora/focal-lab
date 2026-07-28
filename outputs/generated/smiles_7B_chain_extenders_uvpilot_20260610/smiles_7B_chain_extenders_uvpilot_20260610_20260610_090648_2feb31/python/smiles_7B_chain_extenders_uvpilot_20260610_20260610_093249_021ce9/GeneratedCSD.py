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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one valid SMILES string for a chain_extenders molecule. Chain extenders are small bifunctional molecules (MW < 500) used in polyurethane synthesis, typically with two -OH or -NH2 groups. Generate a novel chain extender SMILES. Output ONLY the SMILES string.")))
        if ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
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
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    d_5_spanLen_: int
                    d_5_spanLen_ = len(currentConstrainedOut)
                    d_6_isComplete_: bool
                    d_6_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                    if (d_6_isComplete_) and ((d_5_spanLen_) >= (4)):
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
                        generated = d_7_cg_
                        insideConstrainedOut = d_8_ci_
                        currentConstrainedOut = d_9_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif ((d_6_isComplete_) and ((d_5_spanLen_) >= (1))) and (((d_1_steps_) + (2)) >= (maxSteps)):
                        d_10_cg_: _dafny.Seq
                        d_11_ci_: bool
                        d_12_cc_: _dafny.Seq
                        out6_: _dafny.Seq
                        out7_: bool
                        out8_: _dafny.Seq
                        out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_10_cg_ = out6_
                        d_11_ci_ = out7_
                        d_12_cc_ = out8_
                        generated = d_10_cg_
                        insideConstrainedOut = d_11_ci_
                        currentConstrainedOut = d_12_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        if not(d_6_isComplete_):
                            d_13_constrainedPrompt_: _dafny.Seq
                            d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (d_5_spanLen_):]))
                            d_14_next_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 20, eosToken)
                            d_14_next_ = out9_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_14_next_) == (eosToken):
                                d_15_rg_: _dafny.Seq
                                d_16_rc_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: _dafny.Seq
                                out10_, out11_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_15_rg_ = out10_
                                d_16_rc_ = out11_
                                generated = d_15_rg_
                                currentConstrainedOut = d_16_rc_
                                if ((d_1_steps_) < (maxSteps)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                                    d_17_cg_: _dafny.Seq
                                    d_18_ci_: bool
                                    d_19_cc_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_17_cg_ = out12_
                                    d_18_ci_ = out13_
                                    d_19_cc_ = out14_
                                    generated = d_17_cg_
                                    insideConstrainedOut = d_18_ci_
                                    currentConstrainedOut = d_19_cc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                d_20_stillNotComplete_: bool
                                d_20_stillNotComplete_ = not((parser).IsCompletePrefix(currentConstrainedOut))
                                if d_20_stillNotComplete_:
                                    d_21_ag_: _dafny.Seq
                                    d_22_ai_: bool
                                    d_23_ac_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                    d_21_ag_ = out15_
                                    d_22_ai_ = out16_
                                    d_23_ac_ = out17_
                                    generated = d_21_ag_
                                    insideConstrainedOut = d_22_ai_
                                    currentConstrainedOut = d_23_ac_
                        elif True:
                            d_24_cg_: _dafny.Seq
                            d_25_ci_: bool
                            d_26_cc_: _dafny.Seq
                            out18_: _dafny.Seq
                            out19_: bool
                            out20_: _dafny.Seq
                            out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_24_cg_ = out18_
                            d_25_ci_ = out19_
                            d_26_cc_ = out20_
                            generated = d_24_cg_
                            insideConstrainedOut = d_25_ci_
                            currentConstrainedOut = d_26_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                    pass
            pass
        if insideConstrainedOut:
            d_27_rg_: _dafny.Seq
            d_28_rc_: _dafny.Seq
            out21_: _dafny.Seq
            out22_: _dafny.Seq
            out21_, out22_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_27_rg_ = out21_
            d_28_rc_ = out22_
            generated = d_27_rg_
            currentConstrainedOut = d_28_rc_
            if ((d_1_steps_) < (maxSteps)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                d_29_cg_: _dafny.Seq
                d_30_ci_: bool
                d_31_cc_: _dafny.Seq
                out23_: _dafny.Seq
                out24_: bool
                out25_: _dafny.Seq
                out23_, out24_, out25_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_29_cg_ = out23_
                d_30_ci_ = out24_
                d_31_cc_ = out25_
                generated = d_29_cg_
                insideConstrainedOut = d_30_ci_
                currentConstrainedOut = d_31_cc_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

