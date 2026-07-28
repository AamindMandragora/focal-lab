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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. Write all reasoning first using variable names from the problem. At the VERY END, place the final symbolic expression inside EXACTLY ONE pair of << >>. Use ONLY variable names and operators: +, -, *, /, //, %, (, ), int(). Use int() whenever the result must be a whole number (e.g., counts, discrete items). NEVER use { } braces inside << >>. NEVER use ** for exponentiation. NEVER place text inside << >>. The LAST << >> you write is the answer. Keep the final expression compact and correct.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanEverOpened_: bool
        d_2_spanEverOpened_ = insideConstrained
        d_3_finalModeDone_: bool
        d_3_finalModeDone_ = False
        d_4_reserved_: int = int(0)
        d_5_fracReserve_: int
        d_5_fracReserve_ = _dafny.euclidian_division((maxSteps) * (35), 100)
        if (d_5_fracReserve_) >= (80):
            d_4_reserved_ = d_5_fracReserve_
        elif (maxSteps) >= (80):
            d_4_reserved_ = 80
        elif True:
            d_4_reserved_ = _dafny.euclidian_division(maxSteps, 2)
        if (d_4_reserved_) >= (maxSteps):
            d_4_reserved_ = _dafny.euclidian_division(maxSteps, 2)
        d_6_forceOpenAt_: int
        d_6_forceOpenAt_ = (maxSteps) - (d_4_reserved_)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if d_3_finalModeDone_:
                        raise _dafny.Break("0")
                    if not(insideConstrainedOut):
                        if (d_1_steps_) >= (d_6_forceOpenAt_):
                            d_7_og_: _dafny.Seq
                            d_8_oi_: bool
                            d_9_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_7_og_ = out0_
                            d_8_oi_ = out1_
                            d_9_oc_ = out2_
                            generated = d_7_og_
                            insideConstrainedOut = d_8_oi_
                            currentConstrainedOut = d_9_oc_
                            d_2_spanEverOpened_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_10_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_10_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_10_next_) == (eosToken):
                                if (d_1_steps_) < (maxSteps):
                                    d_11_og_: _dafny.Seq
                                    d_12_oi_: bool
                                    d_13_oc_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_11_og_ = out4_
                                    d_12_oi_ = out5_
                                    d_13_oc_ = out6_
                                    generated = d_11_og_
                                    insideConstrainedOut = d_12_oi_
                                    currentConstrainedOut = d_13_oc_
                                    d_2_spanEverOpened_ = True
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                                if (d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    generated = out7_
                                    insideConstrainedOut = out8_
                                    currentConstrainedOut = out9_
                                    d_2_spanEverOpened_ = True
                    elif True:
                        d_14_cg_: _dafny.Seq
                        d_15_ci_: bool
                        d_16_cc_: _dafny.Seq
                        d_17_closed_: bool
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out13_: bool
                        out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_14_cg_ = out10_
                        d_15_ci_ = out11_
                        d_16_cc_ = out12_
                        d_17_closed_ = out13_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_17_closed_:
                            generated = d_14_cg_
                            insideConstrainedOut = d_15_ci_
                            currentConstrainedOut = d_16_cc_
                            if (d_1_steps_) > (d_6_forceOpenAt_):
                                d_3_finalModeDone_ = True
                        elif True:
                            d_18_constrainedPrompt_: _dafny.Seq
                            d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_19_next_: _dafny.Seq
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_19_next_ = out14_
                            if (d_19_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_20_ag_: _dafny.Seq
                                d_21_ai_: bool
                                d_22_ac_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                d_20_ag_ = out15_
                                d_21_ai_ = out16_
                                d_22_ac_ = out17_
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
            out18_: _dafny.Seq
            out19_: bool
            out20_: _dafny.Seq
            out18_, out19_, out20_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_23_closeBudget_)
            d_24_cg_ = out18_
            d_25_ci_ = out19_
            d_26_cc_ = out20_
            generated = d_24_cg_
            insideConstrainedOut = d_25_ci_
            currentConstrainedOut = d_26_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

