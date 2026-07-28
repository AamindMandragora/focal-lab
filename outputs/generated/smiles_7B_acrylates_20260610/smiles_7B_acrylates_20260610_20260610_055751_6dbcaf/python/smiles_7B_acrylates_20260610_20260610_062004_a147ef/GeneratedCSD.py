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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You are generating a SMILES string token by token for an acrylate molecule. An acrylate MUST contain the vinyl ester group C=CC(=O)O. Start with C= as the first two tokens to avoid generating a trivially complete single-atom SMILES. Generate EXACTLY this acrylate SMILES: C=CC(=O)OCC (methyl acrylate ethyl ester). Do NOT generate just 'C' alone."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_penaltyTokens_: _dafny.Seq
        d_3_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "S")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "B")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "F")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "I")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "P"))])
        if ((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_4_og_: _dafny.Seq
            d_5_oi_: bool
            d_6_oc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_4_og_ = out0_
            d_5_oi_ = out1_
            d_6_oc_ = out2_
            generated = d_4_og_
            insideConstrainedOut = d_5_oi_
            currentConstrainedOut = d_6_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    elif True:
                        d_7_isComplete_: bool
                        d_7_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        d_8_tokenCount_: int
                        d_8_tokenCount_ = len(currentConstrainedOut)
                        if (d_7_isComplete_) and ((d_8_tokenCount_) >= (6)):
                            d_9_cg_: _dafny.Seq
                            d_10_ci_: bool
                            d_11_cc_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_9_cg_ = out3_
                            d_10_ci_ = out4_
                            d_11_cc_ = out5_
                            generated = d_9_cg_
                            insideConstrainedOut = d_10_ci_
                            currentConstrainedOut = d_11_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            raise _dafny.Break("0")
                        elif d_7_isComplete_:
                            d_12_cg_: _dafny.Seq
                            d_13_ci_: bool
                            d_14_cc_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_12_cg_ = out6_
                            d_13_ci_ = out7_
                            d_14_cc_ = out8_
                            generated = d_12_cg_
                            insideConstrainedOut = d_13_ci_
                            currentConstrainedOut = d_14_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_15_stableLen_: int
                            d_15_stableLen_ = (len(generated)) - (len(currentConstrainedOut))
                            d_16_constrainedPrompt_: _dafny.Seq
                            d_16_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:d_15_stableLen_:]))
                            d_17_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            if (d_8_tokenCount_) == (0):
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), d_3_penaltyTokens_, _dafny.BigRational('8e0'), 20, eosToken)
                                d_17_next_ = out9_
                            elif True:
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), 16, eosToken)
                                d_17_next_ = out10_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_17_next_) == (eosToken):
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                    d_18_cg_: _dafny.Seq
                                    d_19_ci_: bool
                                    d_20_cc_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_18_cg_ = out11_
                                    d_19_ci_ = out12_
                                    d_20_cc_ = out13_
                                    generated = d_18_cg_
                                    insideConstrainedOut = d_19_ci_
                                    currentConstrainedOut = d_20_cc_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                d_21_ag_: _dafny.Seq
                                d_22_ai_: bool
                                d_23_ac_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                d_21_ag_ = out14_
                                d_22_ai_ = out15_
                                d_23_ac_ = out16_
                                generated = d_21_ag_
                                insideConstrainedOut = d_22_ai_
                                currentConstrainedOut = d_23_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

