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
        d_1_steps_: int
        d_1_steps_ = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate ONE valid SMILES string for a chain_extender molecule. Chain extenders are bifunctional molecules with exactly two reactive groups. Valid reactive group combinations: two hydroxyl groups (diol), two primary amine groups (diamine), or one hydroxyl and one primary amine (amino alcohol). Consider varied carbon chain lengths (3-10 carbons), branching patterns, cyclic rings (cyclohexane, benzene), ether oxygens in the chain, and secondary functional groups. Generate a chemically valid SMILES string only - no other text.")))
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
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (3)):
                        if (d_1_steps_) < (maxSteps):
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
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        d_8_constrainedPrompt_: _dafny.Seq
                        d_8_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_9_seqLen_: int
                        d_9_seqLen_ = len(currentConstrainedOut)
                        d_10_next_: _dafny.Seq
                        d_10_next_ = eosToken
                        if (d_9_seqLen_) == (0):
                            d_11_softNext_: _dafny.Seq
                            d_12___v0_: bool
                            out6_: _dafny.Seq
                            out7_: bool
                            out6_, out7_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('5e0'), eosToken)
                            d_11_softNext_ = out6_
                            d_12___v0_ = out7_
                            d_10_next_ = d_11_softNext_
                        elif (d_9_seqLen_) == (1):
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('4e0'), eosToken)
                            d_10_next_ = out8_
                        elif (d_9_seqLen_) == (2):
                            d_13_softNext2_: _dafny.Seq
                            d_14___v1_: bool
                            out9_: _dafny.Seq
                            out10_: bool
                            out9_, out10_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('5e0'), eosToken)
                            d_13_softNext2_ = out9_
                            d_14___v1_ = out10_
                            d_10_next_ = d_13_softNext2_
                        elif (d_9_seqLen_) == (3):
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('35e-1'), eosToken)
                            d_10_next_ = out11_
                        elif (d_9_seqLen_) <= (8):
                            if (_dafny.euclidian_modulus(d_9_seqLen_, 2)) == (0):
                                d_15_softNext3_: _dafny.Seq
                                d_16___v2_: bool
                                out12_: _dafny.Seq
                                out13_: bool
                                out12_, out13_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('5e0'), eosToken)
                                d_15_softNext3_ = out12_
                                d_16___v2_ = out13_
                                d_10_next_ = d_15_softNext3_
                            elif True:
                                out14_: _dafny.Seq
                                out14_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('3e0'), eosToken)
                                d_10_next_ = out14_
                        elif (d_9_seqLen_) <= (16):
                            if (_dafny.euclidian_modulus(d_9_seqLen_, 3)) == (0):
                                d_17_softNext4_: _dafny.Seq
                                d_18___v3_: bool
                                out15_: _dafny.Seq
                                out16_: bool
                                out15_, out16_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('4e0'), eosToken)
                                d_17_softNext4_ = out15_
                                d_18___v3_ = out16_
                                d_10_next_ = d_17_softNext4_
                            elif (_dafny.euclidian_modulus(d_9_seqLen_, 3)) == (1):
                                out17_: _dafny.Seq
                                out17_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('25e-1'), eosToken)
                                d_10_next_ = out17_
                            elif True:
                                out18_: _dafny.Seq
                                out18_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_10_next_ = out18_
                        elif True:
                            out19_: _dafny.Seq
                            out19_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_10_next_ = out19_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_10_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_19_ag_: _dafny.Seq
                            d_20_ai_: bool
                            d_21_ac_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: bool
                            out22_: _dafny.Seq
                            out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                            d_19_ag_ = out20_
                            d_20_ai_ = out21_
                            d_21_ac_ = out22_
                            generated = d_19_ag_
                            insideConstrainedOut = d_20_ai_
                            currentConstrainedOut = d_21_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_22_closeBudget_: int
            d_22_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_23_cg3_: _dafny.Seq
            d_24_ci3_: bool
            d_25_cc3_: _dafny.Seq
            out23_: _dafny.Seq
            out24_: bool
            out25_: _dafny.Seq
            out23_, out24_, out25_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_22_closeBudget_)
            d_23_cg3_ = out23_
            d_24_ci3_ = out24_
            d_25_cc3_ = out25_
            generated = d_23_cg3_
            insideConstrainedOut = d_24_ci3_
            currentConstrainedOut = d_25_cc3_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

