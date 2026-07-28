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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate ONE valid SMILES for a chain_extender. Chain extenders MUST have exactly two functional groups (two OH, two NH2, or one OH + one NH2). Do NOT generate molecules with only one functional group. Examples of valid chain extenders: NCCO, NCCCO, NCCCCN, NCCCCCN, OCC(C)CO, OCCO, OCCCO, OCCCCO, OCCOCCO, OCC(CO)CO, NCCNCC, NC1CCCCC1N, NCCCC(N)CC, OCC(CC)CO, OCCCC(O)CC, NC(CC)CN, OCCC(O)CCC, NC(C)(C)CN. Output ONLY the SMILES string, nothing else.")))
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
                    elif ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_2_minSpanLength_)):
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
                        (d_0_helpers_).ScaleAllLogits(lm, _dafny.BigRational('1e-1'))
                        d_10_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        d_11_spanLen_: int
                        d_11_spanLen_ = len(currentConstrainedOut)
                        if (d_11_spanLen_) == (0):
                            d_12_ns_: _dafny.Seq
                            d_13_ok_: bool
                            out6_: _dafny.Seq
                            out7_: bool
                            out6_, out7_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('12e0'), eosToken)
                            d_12_ns_ = out6_
                            d_13_ok_ = out7_
                            d_10_next_ = d_12_ns_
                        elif (d_11_spanLen_) < (3):
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('1e1'), 20, eosToken)
                            d_10_next_ = out8_
                        elif (d_11_spanLen_) < (8):
                            if (_dafny.euclidian_modulus(d_11_spanLen_, 3)) == (0):
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('5e0'), eosToken)
                                d_10_next_ = out9_
                            elif (_dafny.euclidian_modulus(d_11_spanLen_, 3)) == (1):
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('8e0'), 16, eosToken)
                                d_10_next_ = out10_
                            elif True:
                                d_14_ns_: _dafny.Seq
                                d_15_ok_: bool
                                out11_: _dafny.Seq
                                out12_: bool
                                out11_, out12_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('1e1'), eosToken)
                                d_14_ns_ = out11_
                                d_15_ok_ = out12_
                                d_10_next_ = d_14_ns_
                        elif True:
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('4e0'), eosToken)
                            d_10_next_ = out13_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_10_next_) == (eosToken):
                            if (((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_2_minSpanLength_))) and ((d_1_steps_) < (maxSteps)):
                                d_16_cg_: _dafny.Seq
                                d_17_ci_: bool
                                d_18_cc_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_16_cg_ = out14_
                                d_17_ci_ = out15_
                                d_18_cc_ = out16_
                                generated = d_16_cg_
                                insideConstrainedOut = d_17_ci_
                                currentConstrainedOut = d_18_cc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_19_ag_: _dafny.Seq
                            d_20_ai_: bool
                            d_21_ac_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: bool
                            out19_: _dafny.Seq
                            out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                            d_19_ag_ = out17_
                            d_20_ai_ = out18_
                            d_21_ac_ = out19_
                            generated = d_19_ag_
                            insideConstrainedOut = d_20_ai_
                            currentConstrainedOut = d_21_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_22_closeBudget_: int
            d_22_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_23_cg_: _dafny.Seq
            d_24_ci_: bool
            d_25_cc_: _dafny.Seq
            out20_: _dafny.Seq
            out21_: bool
            out22_: _dafny.Seq
            out20_, out21_, out22_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_22_closeBudget_)
            d_23_cg_ = out20_
            d_24_ci_ = out21_
            d_25_cc_ = out22_
            generated = d_23_cg_
            insideConstrainedOut = d_24_ci_
            currentConstrainedOut = d_25_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

