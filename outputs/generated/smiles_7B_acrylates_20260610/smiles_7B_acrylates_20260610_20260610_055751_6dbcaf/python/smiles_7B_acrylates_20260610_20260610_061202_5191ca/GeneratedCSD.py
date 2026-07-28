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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Task: output exactly one SMILES string for a NEW acrylate molecule. The molecule MUST contain the acrylate substructure: C=CC(=O)O (acrylate ester) or C=CC(=O)N (acrylamide). A valid acrylate SMILES must have at least the tokens: C = C C ( = O ) O or similar. Do NOT output just 'C'. Output a complete acrylate ester SMILES such as C=CC(=O)OCC or CCOC(=O)C=C or C=CC(=O)OCCO."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        if ((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_3_og_: _dafny.Seq
            d_4_oi_: bool
            d_5_oc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_3_og_ = out0_
            d_4_oi_ = out1_
            d_5_oc_ = out2_
            generated = d_3_og_
            insideConstrainedOut = d_4_oi_
            currentConstrainedOut = d_5_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    elif True:
                        d_6_isComplete_: bool
                        d_6_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_6_isComplete_:
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
                            d_2_steps_ = (d_2_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_10_stableLen_: int
                            d_10_stableLen_ = (len(generated)) - (len(currentConstrainedOut))
                            d_11_constrainedPrompt_: _dafny.Seq
                            d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:d_10_stableLen_:]))
                            d_12_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), 20, eosToken)
                            d_12_next_ = out6_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_12_next_) == (eosToken):
                                d_13_rg_: _dafny.Seq
                                d_14_rc_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: _dafny.Seq
                                out7_, out8_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_13_rg_ = out7_
                                d_14_rc_ = out8_
                                d_15_rcComplete_: bool
                                d_15_rcComplete_ = (parser).IsCompletePrefix(d_14_rc_)
                                if (d_15_rcComplete_) and ((d_2_steps_) < (maxSteps)):
                                    generated = d_13_rg_
                                    currentConstrainedOut = d_14_rc_
                                    d_16_cg_: _dafny.Seq
                                    d_17_ci_: bool
                                    d_18_cc_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_16_cg_ = out9_
                                    d_17_ci_ = out10_
                                    d_18_cc_ = out11_
                                    generated = d_16_cg_
                                    insideConstrainedOut = d_17_ci_
                                    currentConstrainedOut = d_18_cc_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                elif True:
                                    generated = d_13_rg_
                                    currentConstrainedOut = d_14_rc_
                                raise _dafny.Break("0")
                            elif True:
                                d_19_ag_: _dafny.Seq
                                d_20_ai_: bool
                                d_21_ac_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                                d_19_ag_ = out12_
                                d_20_ai_ = out13_
                                d_21_ac_ = out14_
                                generated = d_19_ag_
                                insideConstrainedOut = d_20_ai_
                                currentConstrainedOut = d_21_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_22_isComplete2_: bool
            d_22_isComplete2_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if d_22_isComplete2_:
                d_23_cg_: _dafny.Seq
                d_24_ci_: bool
                d_25_cc_: _dafny.Seq
                out15_: _dafny.Seq
                out16_: bool
                out17_: _dafny.Seq
                out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_23_cg_ = out15_
                d_24_ci_ = out16_
                d_25_cc_ = out17_
                generated = d_23_cg_
                insideConstrainedOut = d_24_ci_
                currentConstrainedOut = d_25_cc_
                d_2_steps_ = (d_2_steps_) + (1)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

