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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Task: output exactly one valid SMILES for a novel isocyanate molecule. Isocyanates contain the N=C=O group. The SMILES must include the substructure N=C=O (nitrogen double-bonded to carbon double-bonded to oxygen). Examples of valid isocyanate SMILES: O=C=NCC, O=C=NCCC, O=C=NCCCl, O=C=NC(C)C, O=C=NCC(=O)O, O=C=NCCCC. Start with O=C=N and add a carbon chain. Output ONLY the SMILES string."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
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
            d_2_steps_ = (d_2_steps_) + (1)
        d_6_forcedSteps_: int
        d_6_forcedSteps_ = 0
        d_7_minForced_: int
        d_7_minForced_ = 8
        with _dafny.label("0"):
            while (((d_2_steps_) < (maxSteps)) and (insideConstrainedOut)) and ((d_6_forcedSteps_) < (d_7_minForced_)):
                with _dafny.c_label("0"):
                    d_8_constrainedPrompt_: _dafny.Seq
                    d_8_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_9_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('11e-1'), eosToken)
                    d_9_next_ = out3_
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_6_forcedSteps_ = (d_6_forcedSteps_) + (1)
                    if (d_9_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        d_10_isComplete_: bool
                        d_10_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_10_isComplete_:
                            raise _dafny.Break("0")
                        elif True:
                            d_11_ag_: _dafny.Seq
                            d_12_ai_: bool
                            d_13_ac_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                            d_11_ag_ = out4_
                            d_12_ai_ = out5_
                            d_13_ac_ = out6_
                            generated = d_11_ag_
                            insideConstrainedOut = d_12_ai_
                            currentConstrainedOut = d_13_ac_
                    pass
            pass
        with _dafny.label("1"):
            while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    d_14_cg_: _dafny.Seq
                    d_15_ci_: bool
                    d_16_cc_: _dafny.Seq
                    d_17_closed_: bool
                    out7_: _dafny.Seq
                    out8_: bool
                    out9_: _dafny.Seq
                    out10_: bool
                    out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_14_cg_ = out7_
                    d_15_ci_ = out8_
                    d_16_cc_ = out9_
                    d_17_closed_ = out10_
                    d_2_steps_ = (d_2_steps_) + (1)
                    generated = d_14_cg_
                    insideConstrainedOut = d_15_ci_
                    currentConstrainedOut = d_16_cc_
                    if d_17_closed_:
                        raise _dafny.Break("1")
                    elif True:
                        if (d_2_steps_) < (maxSteps):
                            d_18_constrainedPrompt_: _dafny.Seq
                            d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_19_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_19_next_ = out11_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_19_next_) == (eosToken):
                                raise _dafny.Break("1")
                            elif True:
                                d_20_isComplete2_: bool
                                d_20_isComplete2_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if not(d_20_isComplete2_):
                                    d_21_ag_: _dafny.Seq
                                    d_22_ai_: bool
                                    d_23_ac_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                    d_21_ag_ = out12_
                                    d_22_ai_ = out13_
                                    d_23_ac_ = out14_
                                    generated = d_21_ag_
                                    insideConstrainedOut = d_22_ai_
                                    currentConstrainedOut = d_23_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_24_closeBudget_: int
            d_24_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_25_cg_: _dafny.Seq
            d_26_ci_: bool
            d_27_cc_: _dafny.Seq
            out15_: _dafny.Seq
            out16_: bool
            out17_: _dafny.Seq
            out15_, out16_, out17_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_24_closeBudget_)
            d_25_cg_ = out15_
            d_26_ci_ = out16_
            d_27_cc_ = out17_
            generated = d_25_cg_
            insideConstrainedOut = d_26_ci_
            currentConstrainedOut = d_27_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

