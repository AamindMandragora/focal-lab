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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single valid SMILES for a chain_extender. Chain extenders are bifunctional molecules (diol or diamine or amino-alcohol). Output ONLY the SMILES. Prefer unusual/longer structures: OCCCCO, NCCCCN, OCC(C)(C)CO, OCCNCC, NC1CCCCC1N, OCCOCCO, OCC(CO)CO, OCCCO, NCCCO, NC(C)C, OCC(O)CCO, NCCO, NCCNCC, OC(C)CCO.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_minSpanLength_: int
        d_2_minSpanLength_ = 4
        if ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
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
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    elif (((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_2_minSpanLength_))) and (((d_1_steps_) + (1)) <= (maxSteps)):
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
                        (d_0_helpers_).ScaleAllLogits(lm, _dafny.BigRational('15e-2'))
                        d_10_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        d_11_spanLen_: int
                        d_11_spanLen_ = len(currentConstrainedOut)
                        if (d_11_spanLen_) == (0):
                            d_12_ns_: _dafny.Seq
                            d_13_ok_: bool
                            out6_: _dafny.Seq
                            out7_: bool
                            out6_, out7_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('1e1'), eosToken)
                            d_12_ns_ = out6_
                            d_13_ok_ = out7_
                            d_10_next_ = d_12_ns_
                        elif (d_11_spanLen_) == (1):
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('8e0'), 12, eosToken)
                            d_10_next_ = out8_
                        elif (d_11_spanLen_) < (5):
                            if (_dafny.euclidian_modulus(d_11_spanLen_, 2)) == (0):
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('4e0'), eosToken)
                                d_10_next_ = out9_
                            elif True:
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), 16, eosToken)
                                d_10_next_ = out10_
                        elif True:
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('3e0'), eosToken)
                            d_10_next_ = out11_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_10_next_) == (eosToken):
                            if (((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_2_minSpanLength_))) and ((d_1_steps_) < (maxSteps)):
                                d_14_cg_: _dafny.Seq
                                d_15_ci_: bool
                                d_16_cc_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_14_cg_ = out12_
                                d_15_ci_ = out13_
                                d_16_cc_ = out14_
                                generated = d_14_cg_
                                insideConstrainedOut = d_15_ci_
                                currentConstrainedOut = d_16_cc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_17_ag_: _dafny.Seq
                            d_18_ai_: bool
                            d_19_ac_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                            d_17_ag_ = out15_
                            d_18_ai_ = out16_
                            d_19_ac_ = out17_
                            generated = d_17_ag_
                            insideConstrainedOut = d_18_ai_
                            currentConstrainedOut = d_19_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_20_closeBudget_: int
            d_20_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_21_cg_: _dafny.Seq
            d_22_ci_: bool
            d_23_cc_: _dafny.Seq
            out18_: _dafny.Seq
            out19_: bool
            out20_: _dafny.Seq
            out18_, out19_, out20_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_20_closeBudget_)
            d_21_cg_ = out18_
            d_22_ci_ = out19_
            d_23_cc_ = out20_
            generated = d_21_cg_
            insideConstrainedOut = d_22_ci_
            currentConstrainedOut = d_23_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

