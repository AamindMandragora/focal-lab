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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a novel acrylate ester SMILES. The SMILES must contain the acrylate core fragment C=CC(=O)O with a non-trivial ester substituent (at least ethyl or longer). Output only the SMILES. A good acrylate SMILES looks like: C=CC(=O)OCCC or C=CC(=O)OCC(C)C or C=C(C)C(=O)OCCCC. Do not output a single atom. Do not copy examples.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_preambleTarget_: int
        d_2_preambleTarget_ = 5
        with _dafny.label("0"):
            while (((d_1_steps_) < (maxSteps)) and ((d_1_steps_) < (d_2_preambleTarget_))) and (not(insideConstrainedOut)):
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
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_4_og_: _dafny.Seq
            d_5_oi_: bool
            d_6_oc_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_4_og_ = out1_
            d_5_oi_ = out2_
            d_6_oc_ = out3_
            generated = d_4_og_
            insideConstrainedOut = d_5_oi_
            currentConstrainedOut = d_6_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        d_7_minConstrainedTokens_: int
        d_7_minConstrainedTokens_ = 12
        d_8_constrainedTokenCount_: int
        d_8_constrainedTokenCount_ = 0
        with _dafny.label("1"):
            while ((((d_1_steps_) + (2)) < (maxSteps)) and (insideConstrainedOut)) and ((d_8_constrainedTokenCount_) < (d_7_minConstrainedTokens_)):
                with _dafny.c_label("1"):
                    d_9_constrainedPrompt_: _dafny.Seq
                    d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_10_next_: _dafny.Seq
                    out4_: _dafny.Seq
                    out4_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                    d_10_next_ = out4_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_10_next_) == (eosToken):
                        raise _dafny.Break("1")
                    elif True:
                        if not((parser).IsCompletePrefix(currentConstrainedOut)):
                            d_11_ag_: _dafny.Seq
                            d_12_ai_: bool
                            d_13_ac_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                            d_11_ag_ = out5_
                            d_12_ai_ = out6_
                            d_13_ac_ = out7_
                            generated = d_11_ag_
                            insideConstrainedOut = d_12_ai_
                            currentConstrainedOut = d_13_ac_
                            d_8_constrainedTokenCount_ = (d_8_constrainedTokenCount_) + (1)
                        elif True:
                            raise _dafny.Break("1")
                    pass
            pass
        with _dafny.label("2"):
            while (((d_1_steps_) + (2)) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("2"):
                    d_14_cg_: _dafny.Seq
                    d_15_ci_: bool
                    d_16_cc_: _dafny.Seq
                    d_17_closed_: bool
                    out8_: _dafny.Seq
                    out9_: bool
                    out10_: _dafny.Seq
                    out11_: bool
                    out8_, out9_, out10_, out11_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_14_cg_ = out8_
                    d_15_ci_ = out9_
                    d_16_cc_ = out10_
                    d_17_closed_ = out11_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if d_17_closed_:
                        generated = d_14_cg_
                        insideConstrainedOut = d_15_ci_
                        currentConstrainedOut = d_16_cc_
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_next_: _dafny.Seq
                        out12_: _dafny.Seq
                        out12_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_19_next_ = out12_
                        if (d_19_next_) == (eosToken):
                            raise _dafny.Break("2")
                        elif True:
                            if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                d_20_ag_: _dafny.Seq
                                d_21_ai_: bool
                                d_22_ac_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                d_20_ag_ = out13_
                                d_21_ai_ = out14_
                                d_22_ac_ = out15_
                                generated = d_20_ag_
                                insideConstrainedOut = d_21_ai_
                                currentConstrainedOut = d_22_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_23_closeBudget_: int
            d_23_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_24_cg_: _dafny.Seq
            d_25_ci_: bool
            d_26_cc_: _dafny.Seq
            out16_: _dafny.Seq
            out17_: bool
            out18_: _dafny.Seq
            out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_23_closeBudget_)
            d_24_cg_ = out16_
            d_25_ci_ = out17_
            d_26_cc_ = out18_
            generated = d_24_cg_
            insideConstrainedOut = d_25_ci_
            currentConstrainedOut = d_26_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

