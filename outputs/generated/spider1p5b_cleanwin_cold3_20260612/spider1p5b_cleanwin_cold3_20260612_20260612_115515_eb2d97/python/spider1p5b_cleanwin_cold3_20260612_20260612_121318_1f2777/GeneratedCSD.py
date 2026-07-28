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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one SQL query in the format: SQL: <<YOUR QUERY>>. Use only the provided schema. No explanation, no Markdown, no extra text."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_phase_: int
        d_3_phase_ = 0
        d_4_prefixStepLimit_: int
        d_4_prefixStepLimit_ = 6
        d_5_prefixStepsUsed_: int
        d_5_prefixStepsUsed_ = 0
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if (d_3_phase_) == (0):
                        d_6_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_6_next_ = out0_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_5_prefixStepsUsed_ = (d_5_prefixStepsUsed_) + (1)
                        if (d_6_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_3_phase_ = 1
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                            if ((d_5_prefixStepsUsed_) >= (d_4_prefixStepLimit_)) and ((d_2_steps_) < (maxSteps)):
                                d_7_og_: _dafny.Seq
                                d_8_oi_: bool
                                d_9_oc_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_7_og_ = out1_
                                d_8_oi_ = out2_
                                d_9_oc_ = out3_
                                generated = d_7_og_
                                insideConstrainedOut = d_8_oi_
                                currentConstrainedOut = d_9_oc_
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_3_phase_ = 1
                    elif True:
                        if not(insideConstrainedOut):
                            d_3_phase_ = 0
                            d_10_next_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_10_next_ = out4_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_10_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                        elif (parser).IsCompletePrefix(currentConstrainedOut):
                            d_11_cg_: _dafny.Seq
                            d_12_ci_: bool
                            d_13_cc_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_11_cg_ = out5_
                            d_12_ci_ = out6_
                            d_13_cc_ = out7_
                            generated = d_11_cg_
                            insideConstrainedOut = d_12_ci_
                            currentConstrainedOut = d_13_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_3_phase_ = 0
                            raise _dafny.Break("0")
                        elif True:
                            d_14_constrainedPrompt_: _dafny.Seq
                            d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_15_spanLen_: int
                            d_15_spanLen_ = len(currentConstrainedOut)
                            d_16_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            if (d_15_spanLen_) < (10):
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), eosToken)
                                d_16_next_ = out8_
                            elif True:
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_16_next_ = out9_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_16_next_) == (eosToken):
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                    d_17_cg_: _dafny.Seq
                                    d_18_ci_: bool
                                    d_19_cc_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_17_cg_ = out10_
                                    d_18_ci_ = out11_
                                    d_19_cc_ = out12_
                                    generated = d_17_cg_
                                    insideConstrainedOut = d_18_ci_
                                    currentConstrainedOut = d_19_cc_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                d_20_ag_: _dafny.Seq
                                d_21_ai_: bool
                                d_22_ac_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                                d_20_ag_ = out13_
                                d_21_ai_ = out14_
                                d_22_ac_ = out15_
                                generated = d_20_ag_
                                insideConstrainedOut = d_21_ai_
                                currentConstrainedOut = d_22_ac_
                    pass
            pass
        if ((insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_2_steps_) < (maxSteps)):
            d_23_cg_: _dafny.Seq
            d_24_ci_: bool
            d_25_cc_: _dafny.Seq
            out16_: _dafny.Seq
            out17_: bool
            out18_: _dafny.Seq
            out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
            d_23_cg_ = out16_
            d_24_ci_ = out17_
            d_25_cc_ = out18_
            generated = d_23_cg_
            insideConstrainedOut = d_24_ci_
            currentConstrainedOut = d_25_cc_
            d_2_steps_ = (d_2_steps_) + (1)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

