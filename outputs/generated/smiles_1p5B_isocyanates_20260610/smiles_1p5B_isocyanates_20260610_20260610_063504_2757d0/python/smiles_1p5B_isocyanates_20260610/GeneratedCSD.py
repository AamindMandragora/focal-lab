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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one novel SMILES string for the isocyanates class (contains N=C=O group). Output only the SMILES string with no explanation, no Markdown, no exemplar copying."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 12
        d_4_useRepPenalty_: bool
        d_4_useRepPenalty_ = True
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_5_og_: _dafny.Seq
            d_6_oi_: bool
            d_7_oc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_5_og_ = out0_
            d_6_oi_ = out1_
            d_7_oc_ = out2_
            generated = d_5_og_
            insideConstrainedOut = d_6_oi_
            currentConstrainedOut = d_7_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    elif True:
                        d_8_cg_: _dafny.Seq
                        d_9_ci_: bool
                        d_10_cc_: _dafny.Seq
                        d_11_closed_: bool
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out6_: bool
                        out3_, out4_, out5_, out6_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_8_cg_ = out3_
                        d_9_ci_ = out4_
                        d_10_cc_ = out5_
                        d_11_closed_ = out6_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if d_11_closed_:
                            generated = d_8_cg_
                            insideConstrainedOut = d_9_ci_
                            currentConstrainedOut = d_10_cc_
                            raise _dafny.Break("0")
                        elif True:
                            if (d_2_steps_) < (maxSteps):
                                d_12_constrainedPrompt_: _dafny.Seq
                                d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_13_validCount_: int
                                out7_: int
                                out7_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                d_13_validCount_ = out7_
                                d_14_next_: _dafny.Seq
                                d_14_next_ = eosToken
                                if d_4_useRepPenalty_:
                                    out8_: _dafny.Seq
                                    out8_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                    d_14_next_ = out8_
                                elif True:
                                    out9_: _dafny.Seq
                                    out9_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_narrowThreshold_, eosToken)
                                    d_14_next_ = out9_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if (d_14_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_15_ag_: _dafny.Seq
                                    d_16_ai_: bool
                                    d_17_ac_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                    d_15_ag_ = out10_
                                    d_16_ai_ = out11_
                                    d_17_ac_ = out12_
                                    generated = d_15_ag_
                                    insideConstrainedOut = d_16_ai_
                                    currentConstrainedOut = d_17_ac_
                                    d_4_useRepPenalty_ = not(d_4_useRepPenalty_)
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

