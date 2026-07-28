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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a novel acrylate SMILES. Acrylates contain CH2=CH-C(=O)-O or CH2=C(CH3)-C(=O)-O core. Output only the SMILES string.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_minConstrainedTokens_: int
        d_2_minConstrainedTokens_ = 15
        d_3_maxPreamble_: int
        d_3_maxPreamble_ = 80
        d_4_preambleSteps_: int
        d_4_preambleSteps_ = 0
        with _dafny.label("0"):
            while (((d_1_steps_) < (maxSteps)) and ((d_4_preambleSteps_) < (d_3_maxPreamble_))) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_5_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_5_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    d_4_preambleSteps_ = (d_4_preambleSteps_) + (1)
                    if (d_5_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_6_og_: _dafny.Seq
            d_7_oi_: bool
            d_8_oc_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_6_og_ = out1_
            d_7_oi_ = out2_
            d_8_oc_ = out3_
            generated = d_6_og_
            insideConstrainedOut = d_7_oi_
            currentConstrainedOut = d_8_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("1"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("1"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("1")
                    elif True:
                        d_9_remaining_: int
                        d_9_remaining_ = (maxSteps) - (d_1_steps_)
                        if (d_9_remaining_) <= (30):
                            d_10_csg_: _dafny.Seq
                            d_11_csi_: bool
                            d_12_csc_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_9_remaining_)
                            d_10_csg_ = out4_
                            d_11_csi_ = out5_
                            d_12_csc_ = out6_
                            generated = d_10_csg_
                            insideConstrainedOut = d_11_csi_
                            currentConstrainedOut = d_12_csc_
                            d_1_steps_ = maxSteps
                            raise _dafny.Break("1")
                        elif (len(currentConstrainedOut)) >= (d_2_minConstrainedTokens_):
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
                            elif True:
                                if (parser).IsCompletePrefix(currentConstrainedOut):
                                    d_17_cg2_: _dafny.Seq
                                    d_18_ci2_: bool
                                    d_19_cc2_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_17_cg2_ = out11_
                                    d_18_ci2_ = out12_
                                    d_19_cc2_ = out13_
                                    generated = d_17_cg2_
                                    insideConstrainedOut = d_18_ci2_
                                    currentConstrainedOut = d_19_cc2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_20_constrainedPrompt_: _dafny.Seq
                                    d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_21_next_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out14_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e-1'), eosToken)
                                    d_21_next_ = out14_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_21_next_) == (eosToken):
                                        raise _dafny.Break("1")
                                    elif True:
                                        if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                            d_22_ag_: _dafny.Seq
                                            d_23_ai_: bool
                                            d_24_ac_: _dafny.Seq
                                            out15_: _dafny.Seq
                                            out16_: bool
                                            out17_: _dafny.Seq
                                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                            d_22_ag_ = out15_
                                            d_23_ai_ = out16_
                                            d_24_ac_ = out17_
                                            generated = d_22_ag_
                                            insideConstrainedOut = d_23_ai_
                                            currentConstrainedOut = d_24_ac_
                        elif True:
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_25_cg3_: _dafny.Seq
                                d_26_ci3_: bool
                                d_27_cc3_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_25_cg3_ = out18_
                                d_26_ci3_ = out19_
                                d_27_cc3_ = out20_
                                generated = d_25_cg3_
                                insideConstrainedOut = d_26_ci3_
                                currentConstrainedOut = d_27_cc3_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_28_constrainedPrompt_: _dafny.Seq
                                d_28_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_29_next_: _dafny.Seq
                                out21_: _dafny.Seq
                                out21_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_28_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('9e-1'), eosToken)
                                d_29_next_ = out21_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_29_next_) == (eosToken):
                                    raise _dafny.Break("1")
                                elif True:
                                    if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                        d_30_ag2_: _dafny.Seq
                                        d_31_ai2_: bool
                                        d_32_ac2_: _dafny.Seq
                                        out22_: _dafny.Seq
                                        out23_: bool
                                        out24_: _dafny.Seq
                                        out22_, out23_, out24_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_29_next_)
                                        d_30_ag2_ = out22_
                                        d_31_ai2_ = out23_
                                        d_32_ac2_ = out24_
                                        generated = d_30_ag2_
                                        insideConstrainedOut = d_31_ai2_
                                        currentConstrainedOut = d_32_ac2_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_33_remaining2_: int
            d_33_remaining2_ = (maxSteps) - (d_1_steps_)
            d_34_csg2_: _dafny.Seq
            d_35_csi2_: bool
            d_36_csc2_: _dafny.Seq
            out25_: _dafny.Seq
            out26_: bool
            out27_: _dafny.Seq
            out25_, out26_, out27_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_33_remaining2_)
            d_34_csg2_ = out25_
            d_35_csi2_ = out26_
            d_36_csc2_ = out27_
            generated = d_34_csg2_
            insideConstrainedOut = d_35_csi2_
            currentConstrainedOut = d_36_csc2_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

