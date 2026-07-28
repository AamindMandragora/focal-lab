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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one valid novel SMILES string for an isocyanate compound. Isocyanates contain the N=C=O functional group. A minimal isocyanate SMILES is CCN=C=O. Output ONLY the SMILES string, nothing else.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        d_3_minConstrainedLen_: int
        d_3_minConstrainedLen_ = 5
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
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
            d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    if ((len(currentConstrainedOut)) >= (d_3_minConstrainedLen_)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
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
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    if ((d_1_steps_) + (1)) >= (maxSteps):
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_10_cg2_: _dafny.Seq
                            d_11_ci2_: bool
                            d_12_cc2_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_10_cg2_ = out6_
                            d_11_ci2_ = out7_
                            d_12_cc2_ = out8_
                            generated = d_10_cg2_
                            insideConstrainedOut = d_11_ci2_
                            currentConstrainedOut = d_12_cc2_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_13_cg3_: _dafny.Seq
                            d_14_ci3_: bool
                            d_15_cc3_: _dafny.Seq
                            d_16_closed3_: bool
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out12_: bool
                            out9_, out10_, out11_, out12_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_13_cg3_ = out9_
                            d_14_ci3_ = out10_
                            d_15_cc3_ = out11_
                            d_16_closed3_ = out12_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if d_16_closed3_:
                                generated = d_13_cg3_
                                insideConstrainedOut = d_14_ci3_
                                currentConstrainedOut = d_15_cc3_
                        raise _dafny.Break("0")
                    d_17_stableLen_: int
                    d_17_stableLen_ = (len(generated)) - (len(currentConstrainedOut))
                    d_18_constrainedPrompt_: _dafny.Seq
                    d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:d_17_stableLen_:]))
                    d_19_next_: _dafny.Seq
                    out13_: _dafny.Seq
                    out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), d_2_narrowThreshold_, eosToken)
                    d_19_next_ = out13_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_19_next_) == (eosToken):
                        d_20_rg_: _dafny.Seq
                        d_21_rc_: _dafny.Seq
                        out14_: _dafny.Seq
                        out15_: _dafny.Seq
                        out14_, out15_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_20_rg_ = out14_
                        d_21_rc_ = out15_
                        generated = d_20_rg_
                        currentConstrainedOut = d_21_rc_
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_22_cg4_: _dafny.Seq
                            d_23_ci4_: bool
                            d_24_cc4_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_22_cg4_ = out16_
                            d_23_ci4_ = out17_
                            d_24_cc4_ = out18_
                            generated = d_22_cg4_
                            insideConstrainedOut = d_23_ci4_
                            currentConstrainedOut = d_24_cc4_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif not((parser).IsCompletePrefix(currentConstrainedOut)):
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        raise _dafny.Break("0")
                    elif True:
                        d_25_isComplete_: bool
                        d_25_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if not(d_25_isComplete_):
                            d_26_isValid_: bool
                            out19_: bool
                            out19_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_19_next_)
                            d_26_isValid_ = out19_
                            if d_26_isValid_:
                                d_27_ag_: _dafny.Seq
                                d_28_ai_: bool
                                d_29_ac_: _dafny.Seq
                                out20_: _dafny.Seq
                                out21_: bool
                                out22_: _dafny.Seq
                                out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                d_27_ag_ = out20_
                                d_28_ai_ = out21_
                                d_29_ac_ = out22_
                                generated = d_27_ag_
                                insideConstrainedOut = d_28_ai_
                                currentConstrainedOut = d_29_ac_
                        elif True:
                            d_30_cg5_: _dafny.Seq
                            d_31_ci5_: bool
                            d_32_cc5_: _dafny.Seq
                            out23_: _dafny.Seq
                            out24_: bool
                            out25_: _dafny.Seq
                            out23_, out24_, out25_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_30_cg5_ = out23_
                            d_31_ci5_ = out24_
                            d_32_cc5_ = out25_
                            generated = d_30_cg5_
                            insideConstrainedOut = d_31_ci5_
                            currentConstrainedOut = d_32_cc5_
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

