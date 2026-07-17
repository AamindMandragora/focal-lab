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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate ONE valid SMILES string for a chain_extender compound. Chain extenders are bifunctional molecules with exactly two reactive groups: diols (two -OH groups), diamines (two -NH2 groups), or amino alcohols (one -OH and one -NH2). Please generate a STRUCTURALLY DIVERSE chain extender - consider various carbon chain lengths (2 to 10 carbons), branched structures, aromatic rings, aliphatic rings, ether linkages, or combinations. The output must be ONLY the SMILES string.")))
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
                    d_5_cg2_: _dafny.Seq
                    d_6_ci2_: bool
                    d_7_cc2_: _dafny.Seq
                    d_8_closed2_: bool
                    out3_: _dafny.Seq
                    out4_: bool
                    out5_: _dafny.Seq
                    out6_: bool
                    out3_, out4_, out5_, out6_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_5_cg2_ = out3_
                    d_6_ci2_ = out4_
                    d_7_cc2_ = out5_
                    d_8_closed2_ = out6_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if d_8_closed2_:
                        generated = d_5_cg2_
                        insideConstrainedOut = d_6_ci2_
                        currentConstrainedOut = d_7_cc2_
                    elif True:
                        if (d_1_steps_) >= (maxSteps):
                            raise _dafny.Break("0")
                        d_9_constrainedPrompt_: _dafny.Seq
                        d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_10_seqLen_: int
                        d_10_seqLen_ = len(currentConstrainedOut)
                        d_11_next_: _dafny.Seq
                        d_11_next_ = eosToken
                        if ((d_10_seqLen_) < (5)) and (((d_1_steps_) + (2)) <= (maxSteps)):
                            d_12_topCands_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, 3, eosToken)
                            d_12_topCands_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            (d_0_helpers_).SafePenalizeTokenLogits(lm, d_12_topCands_, _dafny.BigRational('6e0'))
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_11_next_ = out8_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_10_seqLen_) < (12):
                            if (_dafny.euclidian_modulus(d_10_seqLen_, 3)) == (0):
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('14e-1'), eosToken)
                                d_11_next_ = out9_
                            elif (_dafny.euclidian_modulus(d_10_seqLen_, 3)) == (1):
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('25e-1'), eosToken)
                                d_11_next_ = out10_
                            elif True:
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_11_next_ = out11_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_11_next_ = out12_
                            d_1_steps_ = (d_1_steps_) + (1)
                        if (d_11_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_13_ag_: _dafny.Seq
                            d_14_ai_: bool
                            d_15_ac_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                            d_13_ag_ = out13_
                            d_14_ai_ = out14_
                            d_15_ac_ = out15_
                            generated = d_13_ag_
                            insideConstrainedOut = d_14_ai_
                            currentConstrainedOut = d_15_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_16_closeBudget_: int
            d_16_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_17_cg3_: _dafny.Seq
            d_18_ci3_: bool
            d_19_cc3_: _dafny.Seq
            out16_: _dafny.Seq
            out17_: bool
            out18_: _dafny.Seq
            out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_16_closeBudget_)
            d_17_cg3_ = out16_
            d_18_ci3_ = out17_
            d_19_cc3_ = out18_
            generated = d_17_cg3_
            insideConstrainedOut = d_18_ci3_
            currentConstrainedOut = d_19_cc3_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

