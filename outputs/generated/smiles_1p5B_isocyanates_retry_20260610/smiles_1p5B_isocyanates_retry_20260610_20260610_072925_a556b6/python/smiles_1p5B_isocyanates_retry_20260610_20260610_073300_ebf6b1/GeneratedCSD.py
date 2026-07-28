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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You are generating SMILES for an isocyanate molecule. Isocyanates contain the N=C=O functional group. You MUST output a complete SMILES with the N=C=O group and an attached carbon chain. Examples of valid isocyanate SMILES: O=C=NC (methyl isocyanate), O=C=NCC (ethyl isocyanate), O=C=NCCC (propyl isocyanate), O=C=NCCCl, O=C=Nc1ccccc1 (phenyl isocyanate). Always start with O=C=N followed by a carbon group. NEVER output a single atom like O or N.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeSteps_: int
        d_2_freeSteps_ = 6
        with _dafny.label("0"):
            while (((d_1_steps_) < (maxSteps)) and ((d_1_steps_) < (d_2_freeSteps_))) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_3_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_3_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_3_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                        if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_4_og2_: _dafny.Seq
                            d_5_oi2_: bool
                            d_6_oc2_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_4_og2_ = out1_
                            d_5_oi2_ = out2_
                            d_6_oc2_ = out3_
                            generated = d_4_og2_
                            insideConstrainedOut = d_5_oi2_
                            currentConstrainedOut = d_6_oc2_
                            raise _dafny.Break("0")
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_7_og_: _dafny.Seq
            d_8_oi_: bool
            d_9_oc_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_7_og_ = out4_
            d_8_oi_ = out5_
            d_9_oc_ = out6_
            generated = d_7_og_
            insideConstrainedOut = d_8_oi_
            currentConstrainedOut = d_9_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        d_10_minConstrainedTokens_: int
        d_10_minConstrainedTokens_ = 5
        d_11_constrainedCount_: int
        d_11_constrainedCount_ = 0
        with _dafny.label("1"):
            while (((d_1_steps_) < (maxSteps)) and (insideConstrainedOut)) and ((d_11_constrainedCount_) < (d_10_minConstrainedTokens_)):
                with _dafny.c_label("1"):
                    d_12_isComp_: bool
                    d_12_isComp_ = (parser).IsCompletePrefix(currentConstrainedOut)
                    if (d_12_isComp_) and ((len(currentConstrainedOut)) < (d_10_minConstrainedTokens_)):
                        d_13_rg_: _dafny.Seq
                        d_14_rc_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: _dafny.Seq
                        out7_, out8_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_13_rg_ = out7_
                        d_14_rc_ = out8_
                        generated = d_13_rg_
                        currentConstrainedOut = d_14_rc_
                        if (d_1_steps_) < (maxSteps):
                            d_15_constrainedPrompt_: _dafny.Seq
                            d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_16_next_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 20, eosToken)
                            d_16_next_ = out9_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_16_next_) == (eosToken):
                                raise _dafny.Break("1")
                            elif True:
                                d_17_isComp2_: bool
                                d_17_isComp2_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if not(d_17_isComp2_):
                                    d_18_ag_: _dafny.Seq
                                    d_19_ai_: bool
                                    d_20_ac_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                                    d_18_ag_ = out10_
                                    d_19_ai_ = out11_
                                    d_20_ac_ = out12_
                                    generated = d_18_ag_
                                    insideConstrainedOut = d_19_ai_
                                    currentConstrainedOut = d_20_ac_
                                    d_11_constrainedCount_ = (d_11_constrainedCount_) + (1)
                    elif d_12_isComp_:
                        raise _dafny.Break("1")
                    elif True:
                        d_21_constrainedPrompt_: _dafny.Seq
                        d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_22_next_: _dafny.Seq
                        out13_: _dafny.Seq
                        out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 20, eosToken)
                        d_22_next_ = out13_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_22_next_) == (eosToken):
                            raise _dafny.Break("1")
                        elif True:
                            d_23_isComp3_: bool
                            d_23_isComp3_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if not(d_23_isComp3_):
                                d_24_ag2_: _dafny.Seq
                                d_25_ai2_: bool
                                d_26_ac2_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                d_24_ag2_ = out14_
                                d_25_ai2_ = out15_
                                d_26_ac2_ = out16_
                                generated = d_24_ag2_
                                insideConstrainedOut = d_25_ai2_
                                currentConstrainedOut = d_26_ac2_
                                d_11_constrainedCount_ = (d_11_constrainedCount_) + (1)
                            elif True:
                                raise _dafny.Break("1")
                    pass
            pass
        with _dafny.label("2"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("2"):
                    d_27_cg_: _dafny.Seq
                    d_28_ci_: bool
                    d_29_cc_: _dafny.Seq
                    d_30_closed_: bool
                    out17_: _dafny.Seq
                    out18_: bool
                    out19_: _dafny.Seq
                    out20_: bool
                    out17_, out18_, out19_, out20_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_27_cg_ = out17_
                    d_28_ci_ = out18_
                    d_29_cc_ = out19_
                    d_30_closed_ = out20_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if d_30_closed_:
                        generated = d_27_cg_
                        insideConstrainedOut = d_28_ci_
                        currentConstrainedOut = d_29_cc_
                        raise _dafny.Break("2")
                    elif True:
                        if (d_1_steps_) < (maxSteps):
                            d_31_isComp4_: bool
                            d_31_isComp4_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_31_isComp4_:
                                raise _dafny.Break("2")
                            elif True:
                                d_32_constrainedPrompt3_: _dafny.Seq
                                d_32_constrainedPrompt3_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_33_next3_: _dafny.Seq
                                out21_: _dafny.Seq
                                out21_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_32_constrainedPrompt3_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_33_next3_ = out21_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_33_next3_) == (eosToken):
                                    raise _dafny.Break("2")
                                elif True:
                                    d_34_isComp5_: bool
                                    d_34_isComp5_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if not(d_34_isComp5_):
                                        d_35_ag3_: _dafny.Seq
                                        d_36_ai3_: bool
                                        d_37_ac3_: _dafny.Seq
                                        out22_: _dafny.Seq
                                        out23_: bool
                                        out24_: _dafny.Seq
                                        out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_33_next3_)
                                        d_35_ag3_ = out22_
                                        d_36_ai3_ = out23_
                                        d_37_ac3_ = out24_
                                        generated = d_35_ag3_
                                        insideConstrainedOut = d_36_ai3_
                                        currentConstrainedOut = d_37_ac3_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_38_isCompFinal_: bool
            d_38_isCompFinal_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if d_38_isCompFinal_:
                d_39_cg_: _dafny.Seq
                d_40_ci_: bool
                d_41_cc_: _dafny.Seq
                out25_: _dafny.Seq
                out26_: bool
                out27_: _dafny.Seq
                out25_, out26_, out27_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_39_cg_ = out25_
                d_40_ci_ = out26_
                d_41_cc_ = out27_
                generated = d_39_cg_
                insideConstrainedOut = d_40_ci_
                currentConstrainedOut = d_41_cc_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

