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
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
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
                        d_9_spanLen_: int
                        d_9_spanLen_ = len(currentConstrainedOut)
                        d_10_constrainedPrompt_: _dafny.Seq
                        d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (d_9_spanLen_):]))
                        d_11_next_: _dafny.Seq
                        out6_: _dafny.Seq
                        out6_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_11_next_ = out6_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_11_next_) == (eosToken):
                            d_12_rg_: _dafny.Seq
                            d_13_rc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: _dafny.Seq
                            out7_, out8_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_12_rg_ = out7_
                            d_13_rc_ = out8_
                            generated = d_12_rg_
                            currentConstrainedOut = d_13_rc_
                            if ((d_1_steps_) < (maxSteps)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
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
                            d_17_ag_: _dafny.Seq
                            d_18_ai_: bool
                            d_19_ac_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                            d_17_ag_ = out12_
                            d_18_ai_ = out13_
                            d_19_ac_ = out14_
                            generated = d_17_ag_
                            insideConstrainedOut = d_18_ai_
                            currentConstrainedOut = d_19_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_20_rg_: _dafny.Seq
            d_21_rc_: _dafny.Seq
            out15_: _dafny.Seq
            out16_: _dafny.Seq
            out15_, out16_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_20_rg_ = out15_
            d_21_rc_ = out16_
            generated = d_20_rg_
            currentConstrainedOut = d_21_rc_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_22_cg_: _dafny.Seq
                d_23_ci_: bool
                d_24_cc_: _dafny.Seq
                out17_: _dafny.Seq
                out18_: bool
                out19_: _dafny.Seq
                out17_, out18_, out19_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_22_cg_ = out17_
                d_23_ci_ = out18_
                d_24_cc_ = out19_
                generated = d_22_cg_
                insideConstrainedOut = d_23_ci_
                currentConstrainedOut = d_24_cc_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

