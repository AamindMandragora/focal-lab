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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a novel acrylate SMILES. Must contain C=CC(=O)O core. Examples of valid acrylate SMILES: C=CC(=O)OCCC, C=CC(=O)OCCCC, C=CC(=O)OCC(C)C, C=C(C)C(=O)OCCC, C=CC(=O)OCCCCC, C=CC(=O)OCC(CC)CC, C=CC(=O)OCCCCCC, C=CC(=O)OCCOC, C=CC(=O)OCCOCCO, C=CC(=O)OC(C)(C)CC. Output only the SMILES.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_minConstrainedTokens_: int
        d_2_minConstrainedTokens_ = 20
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
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
            d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("0"):
            while (((d_1_steps_) < (maxSteps)) and (insideConstrainedOut)) and ((len(currentConstrainedOut)) < (d_2_minConstrainedTokens_)):
                with _dafny.c_label("0"):
                    d_6_constrainedPrompt_: _dafny.Seq
                    d_6_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_7_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_6_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('11e-1'), eosToken)
                    d_7_next_ = out3_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_7_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        d_8_isComplete_: bool
                        d_8_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if not(d_8_isComplete_):
                            d_9_ag_: _dafny.Seq
                            d_10_ai_: bool
                            d_11_ac_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_7_next_)
                            d_9_ag_ = out4_
                            d_10_ai_ = out5_
                            d_11_ac_ = out6_
                            generated = d_9_ag_
                            insideConstrainedOut = d_10_ai_
                            currentConstrainedOut = d_11_ac_
                    pass
            pass
        with _dafny.label("1"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    d_12_isNowComplete_: bool
                    d_12_isNowComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                    if d_12_isNowComplete_:
                        d_13_cg_: _dafny.Seq
                        d_14_ci_: bool
                        d_15_cc_: _dafny.Seq
                        d_16_closed_: bool
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out10_: bool
                        out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_13_cg_ = out7_
                        d_14_ci_ = out8_
                        d_15_cc_ = out9_
                        d_16_closed_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_16_closed_:
                            generated = d_13_cg_
                            insideConstrainedOut = d_14_ci_
                            currentConstrainedOut = d_15_cc_
                            raise _dafny.Break("1")
                        d_17_constrainedPrompt2_: _dafny.Seq
                        d_17_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_18_next2_: _dafny.Seq
                        out11_: _dafny.Seq
                        out11_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_17_constrainedPrompt2_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), _dafny.SeqWithoutIsStrInference([]), _dafny.BigRational('1e0'), 4, eosToken)
                        d_18_next2_ = out11_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_18_next2_) == (eosToken):
                            raise _dafny.Break("1")
                        elif True:
                            d_19_isComplete3_: bool
                            d_19_isComplete3_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if not(d_19_isComplete3_):
                                d_20_ag2_: _dafny.Seq
                                d_21_ai2_: bool
                                d_22_ac2_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next2_)
                                d_20_ag2_ = out12_
                                d_21_ai2_ = out13_
                                d_22_ac2_ = out14_
                                generated = d_20_ag2_
                                insideConstrainedOut = d_21_ai2_
                                currentConstrainedOut = d_22_ac2_
                    elif True:
                        d_23_constrainedPrompt3_: _dafny.Seq
                        d_23_constrainedPrompt3_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_24_next3_: _dafny.Seq
                        out15_: _dafny.Seq
                        out15_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_23_constrainedPrompt3_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), _dafny.SeqWithoutIsStrInference([]), _dafny.BigRational('1e0'), 4, eosToken)
                        d_24_next3_ = out15_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_24_next3_) == (eosToken):
                            raise _dafny.Break("1")
                        elif True:
                            d_25_isComplete4_: bool
                            d_25_isComplete4_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if not(d_25_isComplete4_):
                                d_26_ag3_: _dafny.Seq
                                d_27_ai3_: bool
                                d_28_ac3_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next3_)
                                d_26_ag3_ = out16_
                                d_27_ai3_ = out17_
                                d_28_ac3_ = out18_
                                generated = d_26_ag3_
                                insideConstrainedOut = d_27_ai3_
                                currentConstrainedOut = d_28_ac3_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_29_closeBudget_: int
            d_29_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_30_cg_: _dafny.Seq
            d_31_ci_: bool
            d_32_cc_: _dafny.Seq
            out19_: _dafny.Seq
            out20_: bool
            out21_: _dafny.Seq
            out19_, out20_, out21_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_29_closeBudget_)
            d_30_cg_ = out19_
            d_31_ci_ = out20_
            d_32_cc_ = out21_
            generated = d_30_cg_
            insideConstrainedOut = d_31_ci_
            currentConstrainedOut = d_32_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

