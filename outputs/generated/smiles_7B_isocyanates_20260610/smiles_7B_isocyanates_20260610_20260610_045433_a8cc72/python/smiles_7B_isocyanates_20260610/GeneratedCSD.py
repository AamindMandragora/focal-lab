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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one valid SMILES string for a novel isocyanate compound containing the N=C=O group. Do not copy any exemplar. Output only the SMILES string with no explanation.")))
        d_1_steps_: int
        d_1_steps_ = 0
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
        d_5_narrowThreshold_: int
        d_5_narrowThreshold_ = 15
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    elif True:
                        d_6_cg_: _dafny.Seq
                        d_7_ci_: bool
                        d_8_cc_: _dafny.Seq
                        d_9_closed_: bool
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out6_: bool
                        out3_, out4_, out5_, out6_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_6_cg_ = out3_
                        d_7_ci_ = out4_
                        d_8_cc_ = out5_
                        d_9_closed_ = out6_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_9_closed_:
                            generated = d_6_cg_
                            insideConstrainedOut = d_7_ci_
                            currentConstrainedOut = d_8_cc_
                            raise _dafny.Break("0")
                        elif True:
                            d_10_stableLen_: int
                            d_10_stableLen_ = (len(generated)) - (len(currentConstrainedOut))
                            d_11_constrainedPrompt_: _dafny.Seq
                            d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:d_10_stableLen_:]))
                            d_12_validCount_: int
                            out7_: int
                            out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_12_validCount_ = out7_
                            d_13_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            if (d_12_validCount_) <= (d_5_narrowThreshold_):
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), eosToken)
                                d_13_next_ = out8_
                            elif True:
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                d_13_next_ = out9_
                            if (d_13_next_) == (eosToken):
                                d_14_rg_: _dafny.Seq
                                d_15_rc_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: _dafny.Seq
                                out10_, out11_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_14_rg_ = out10_
                                d_15_rc_ = out11_
                                generated = d_14_rg_
                                currentConstrainedOut = d_15_rc_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_16_cg2_: _dafny.Seq
                                    d_17_ci2_: bool
                                    d_18_cc2_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_16_cg2_ = out12_
                                    d_17_ci2_ = out13_
                                    d_18_cc2_ = out14_
                                    generated = d_16_cg2_
                                    insideConstrainedOut = d_17_ci2_
                                    currentConstrainedOut = d_18_cc2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                d_19_ag_: _dafny.Seq
                                d_20_ai_: bool
                                d_21_ac_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                                d_19_ag_ = out15_
                                d_20_ai_ = out16_
                                d_21_ac_ = out17_
                                generated = d_19_ag_
                                insideConstrainedOut = d_20_ai_
                                currentConstrainedOut = d_21_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

