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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly: SQL: <<query>> where query is a single valid SQL SELECT statement using only the provided schema. No explanation, no markdown, no extra text.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        d_3_maxFreeSteps_: int
        d_3_maxFreeSteps_ = 30
        d_4_freeSteps_: int
        d_4_freeSteps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_4_freeSteps_) < (d_3_maxFreeSteps_)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                            d_5_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_5_next_ = out0_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_4_freeSteps_ = (d_4_freeSteps_) + (1)
                            if (d_5_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                                if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_6_eg_: _dafny.Seq
                                    d_7_ei_: bool
                                    d_8_ec_: _dafny.Seq
                                    out1_: _dafny.Seq
                                    out2_: bool
                                    out3_: _dafny.Seq
                                    out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_6_eg_ = out1_
                                    d_7_ei_ = out2_
                                    d_8_ec_ = out3_
                                    generated = d_6_eg_
                                    insideConstrainedOut = d_7_ei_
                                    currentConstrainedOut = d_8_ec_
                        elif True:
                            if ((d_1_steps_) + (1)) <= (maxSteps):
                                d_9_og_: _dafny.Seq
                                d_10_oi_: bool
                                d_11_oc_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_9_og_ = out4_
                                d_10_oi_ = out5_
                                d_11_oc_ = out6_
                                generated = d_9_og_
                                insideConstrainedOut = d_10_oi_
                                currentConstrainedOut = d_11_oc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        if ((d_1_steps_) + (1)) <= (maxSteps):
                            d_12_cg_: _dafny.Seq
                            d_13_ci_: bool
                            d_14_cc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_12_cg_ = out7_
                            d_13_ci_ = out8_
                            d_14_cc_ = out9_
                            generated = d_12_cg_
                            insideConstrainedOut = d_13_ci_
                            currentConstrainedOut = d_14_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_16_validCount_: int
                        out10_: int
                        out10_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_16_validCount_ = out10_
                        d_17_next_: _dafny.Seq
                        d_17_next_ = eosToken
                        if (d_16_validCount_) <= (d_2_narrowThreshold_):
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), eosToken)
                            d_17_next_ = out11_
                        elif True:
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                            d_17_next_ = out12_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_17_next_) == (eosToken):
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                d_18_cg_: _dafny.Seq
                                d_19_ci_: bool
                                d_20_cc_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_18_cg_ = out13_
                                d_19_ci_ = out14_
                                d_20_cc_ = out15_
                                generated = d_18_cg_
                                insideConstrainedOut = d_19_ci_
                                currentConstrainedOut = d_20_cc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_21_ag_: _dafny.Seq
                            d_22_ai_: bool
                            d_23_ac_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                            d_21_ag_ = out16_
                            d_22_ai_ = out17_
                            d_23_ac_ = out18_
                            generated = d_21_ag_
                            insideConstrainedOut = d_22_ai_
                            currentConstrainedOut = d_23_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

