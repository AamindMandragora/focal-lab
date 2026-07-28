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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a SMILES string for the isocyanates class. Isocyanates contain the N=C=O group. Example valid isocyanate SMILES: CN=C=O, CCN=C=O, O=C=NCCl, O=C=Nc1ccccc1. Generate a novel isocyanate SMILES containing N=C=O. The molecule should have at least 5 heavy atoms.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
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
                    elif True:
                        d_5_cg_: _dafny.Seq
                        d_6_ci_: bool
                        d_7_cc_: _dafny.Seq
                        d_8_closed_: bool
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out6_: bool
                        out3_, out4_, out5_, out6_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_5_cg_ = out3_
                        d_6_ci_ = out4_
                        d_7_cc_ = out5_
                        d_8_closed_ = out6_
                        if d_8_closed_:
                            d_1_steps_ = (d_1_steps_) + (1)
                            generated = d_5_cg_
                            insideConstrainedOut = d_6_ci_
                            currentConstrainedOut = d_7_cc_
                        elif True:
                            d_9_constrainedPrompt_: _dafny.Seq
                            d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_10_next_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_10_next_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_10_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_11_isComplete_: bool
                                d_11_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if d_11_isComplete_:
                                    raise _dafny.Break("0")
                                elif True:
                                    d_12_ag_: _dafny.Seq
                                    d_13_ai_: bool
                                    d_14_ac_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out9_: bool
                                    out10_: _dafny.Seq
                                    out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                                    d_12_ag_ = out8_
                                    d_13_ai_ = out9_
                                    d_14_ac_ = out10_
                                    generated = d_12_ag_
                                    insideConstrainedOut = d_13_ai_
                                    currentConstrainedOut = d_14_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_15_isComp_: bool
            d_15_isComp_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if d_15_isComp_:
                d_16_cg_: _dafny.Seq
                d_17_ci_: bool
                d_18_cc_: _dafny.Seq
                out11_: _dafny.Seq
                out12_: bool
                out13_: _dafny.Seq
                out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_16_cg_ = out11_
                d_17_ci_ = out12_
                d_18_cc_ = out13_
                generated = d_16_cg_
                insideConstrainedOut = d_17_ci_
                currentConstrainedOut = d_18_cc_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

