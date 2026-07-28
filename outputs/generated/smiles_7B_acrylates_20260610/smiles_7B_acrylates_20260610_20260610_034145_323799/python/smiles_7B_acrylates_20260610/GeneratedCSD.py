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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one novel, valid SMILES string for the acrylates class. Output only the SMILES string and nothing else. Do not copy any exemplar from the prompt. The SMILES must contain the acrylate functional group (C=C-C(=O)-O)."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
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
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
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
                    elif True:
                        d_10_constrainedPrompt_: _dafny.Seq
                        d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_11_validCount_: int
                        out6_: int
                        out6_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_11_validCount_ = out6_
                        d_12_next_: _dafny.Seq
                        d_12_next_ = eosToken
                        if (d_11_validCount_) <= (d_3_narrowThreshold_):
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), d_3_narrowThreshold_, eosToken)
                            d_12_next_ = out7_
                        elif True:
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_12_next_ = out8_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_12_next_) == (eosToken):
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                pass
                            elif True:
                                d_13_rg_: _dafny.Seq
                                d_14_rc_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: _dafny.Seq
                                out9_, out10_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_13_rg_ = out9_
                                d_14_rc_ = out10_
                                generated = d_13_rg_
                                currentConstrainedOut = d_14_rc_
                            raise _dafny.Break("0")
                        elif True:
                            d_15_ag_: _dafny.Seq
                            d_16_ai_: bool
                            d_17_ac_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                            d_15_ag_ = out11_
                            d_16_ai_ = out12_
                            d_17_ac_ = out13_
                            generated = d_15_ag_
                            insideConstrainedOut = d_16_ai_
                            currentConstrainedOut = d_17_ac_
                    pass
            pass
        if ((insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_2_steps_) < (maxSteps)):
            d_18_cg_: _dafny.Seq
            d_19_ci_: bool
            d_20_cc_: _dafny.Seq
            out14_: _dafny.Seq
            out15_: bool
            out16_: _dafny.Seq
            out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
            d_18_cg_ = out14_
            d_19_ci_ = out15_
            d_20_cc_ = out16_
            generated = d_18_cg_
            insideConstrainedOut = d_19_ci_
            currentConstrainedOut = d_20_cc_
            d_2_steps_ = (d_2_steps_) + (1)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

