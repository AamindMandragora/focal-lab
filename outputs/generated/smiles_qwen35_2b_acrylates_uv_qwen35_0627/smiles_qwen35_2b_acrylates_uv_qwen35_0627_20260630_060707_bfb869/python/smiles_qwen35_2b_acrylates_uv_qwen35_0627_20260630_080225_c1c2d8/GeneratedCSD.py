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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output ONE valid acrylate SMILES and nothing else. An acrylate must contain the vinyl ester fragment C=CC(=O)O followed by an alkyl or substituted alkyl group. Valid acrylates: C=CC(=O)OCC, C=CC(=O)OCCC, C=CC(=O)OC(C)C, C=CC(=O)OCCCC, C=CC(=O)OCCO, C=CC(=O)OCCCCC, C=CC(=O)OCC(C)C, C=CC(=O)OC(C)(C)C, C=CC(=O)OCCOC, C=CC(=O)OCCCOC, C=CC(=O)OCC(O)C, C=CC(=O)OCCN(C)C. Stop generating as soon as the SMILES is complete. Do NOT add extra atoms after the ester is complete.")))
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
        d_5_innerSteps_: int
        d_5_innerSteps_ = 0
        d_6_maxInnerSteps_: int
        d_6_maxInnerSteps_ = 60
        with _dafny.label("0"):
            while ((insideConstrainedOut) and ((d_1_steps_) < (maxSteps))) and ((d_5_innerSteps_) < (d_6_maxInnerSteps_)):
                with _dafny.c_label("0"):
                    d_7_cg_: _dafny.Seq
                    d_8_ci_: bool
                    d_9_cc_: _dafny.Seq
                    d_10_closed_: bool
                    out3_: _dafny.Seq
                    out4_: bool
                    out5_: _dafny.Seq
                    out6_: bool
                    out3_, out4_, out5_, out6_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                    d_7_cg_ = out3_
                    d_8_ci_ = out4_
                    d_9_cc_ = out5_
                    d_10_closed_ = out6_
                    if d_10_closed_:
                        generated = d_7_cg_
                        insideConstrainedOut = d_8_ci_
                        currentConstrainedOut = d_9_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_5_innerSteps_ = (d_5_innerSteps_) + (1)
                    elif True:
                        if (d_1_steps_) < (maxSteps):
                            d_11_constrainedPrompt_: _dafny.Seq
                            d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_12_next_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('12e-1'), eosToken)
                            d_12_next_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_5_innerSteps_ = (d_5_innerSteps_) + (1)
                            if (d_12_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_13_appendedGenerated_: _dafny.Seq
                                d_14_appendedInside_: bool
                                d_15_appendedCurrent_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                                d_13_appendedGenerated_ = out8_
                                d_14_appendedInside_ = out9_
                                d_15_appendedCurrent_ = out10_
                                generated = d_13_appendedGenerated_
                                insideConstrainedOut = d_14_appendedInside_
                                currentConstrainedOut = d_15_appendedCurrent_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_16_remaining_: int
            d_16_remaining_ = (maxSteps) - (d_1_steps_)
            d_17_closeBudget_: int
            if (d_16_remaining_) < (30):
                d_17_closeBudget_ = d_16_remaining_
            elif True:
                d_17_closeBudget_ = 30
            d_18_cg2_: _dafny.Seq
            d_19_ci2_: bool
            d_20_cc2_: _dafny.Seq
            out11_: _dafny.Seq
            out12_: bool
            out13_: _dafny.Seq
            out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_closeBudget_)
            d_18_cg2_ = out11_
            d_19_ci2_ = out12_
            d_20_cc2_ = out13_
            generated = d_18_cg2_
            insideConstrainedOut = d_19_ci2_
            currentConstrainedOut = d_20_cc2_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

