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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. Write all reasoning first using variable names from the problem. At the very end, place the final symbolic expression inside EXACTLY ONE pair of << >>. Use ONLY variable names and operators: +, -, *, /, //, %, (, ), int(). Use int() when the answer must be a whole number. NEVER use { } braces inside << >>. NEVER use ** for exponentiation. NEVER place text inside << >>. Write ONLY ONE << >> pair at the very end.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanEverOpened_: bool
        d_2_spanEverOpened_ = insideConstrained
        d_3_finalMode_: bool
        d_3_finalMode_ = False
        d_4_breakAfterClose_: bool
        d_4_breakAfterClose_ = False
        d_5_reserved_: int = int(0)
        d_6_fracReserve_: int
        d_6_fracReserve_ = _dafny.euclidian_division((maxSteps) * (35), 100)
        if (d_6_fracReserve_) >= (80):
            d_5_reserved_ = d_6_fracReserve_
        elif (maxSteps) >= (80):
            d_5_reserved_ = 80
        elif True:
            d_5_reserved_ = _dafny.euclidian_division(maxSteps, 2)
        if (d_5_reserved_) >= (maxSteps):
            d_5_reserved_ = _dafny.euclidian_division(maxSteps, 2)
        d_7_forceOpenAt_: int
        d_7_forceOpenAt_ = (maxSteps) - (d_5_reserved_)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_2_spanEverOpened_)) and ((d_1_steps_) >= (d_7_forceOpenAt_)):
                            d_8_og_: _dafny.Seq
                            d_9_oi_: bool
                            d_10_oc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_8_og_ = out0_
                            d_9_oi_ = out1_
                            d_10_oc_ = out2_
                            generated = d_8_og_
                            insideConstrainedOut = d_9_oi_
                            currentConstrainedOut = d_10_oc_
                            d_2_spanEverOpened_ = True
                            d_3_finalMode_ = True
                            d_4_breakAfterClose_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif ((d_2_spanEverOpened_) and ((d_1_steps_) >= (d_7_forceOpenAt_))) and (not(d_3_finalMode_)):
                            d_11_og_: _dafny.Seq
                            d_12_oi_: bool
                            d_13_oc_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_11_og_ = out3_
                            d_12_oi_ = out4_
                            d_13_oc_ = out5_
                            generated = d_11_og_
                            insideConstrainedOut = d_12_oi_
                            currentConstrainedOut = d_13_oc_
                            d_3_finalMode_ = True
                            d_4_breakAfterClose_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif d_4_breakAfterClose_:
                            raise _dafny.Break("0")
                        elif True:
                            d_14_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_14_next_ = out6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_14_next_) == (eosToken):
                                if (not(d_2_spanEverOpened_)) and ((d_1_steps_) < (maxSteps)):
                                    d_15_og_: _dafny.Seq
                                    d_16_oi_: bool
                                    d_17_oc_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_15_og_ = out7_
                                    d_16_oi_ = out8_
                                    d_17_oc_ = out9_
                                    generated = d_15_og_
                                    insideConstrainedOut = d_16_oi_
                                    currentConstrainedOut = d_17_oc_
                                    d_2_spanEverOpened_ = True
                                    d_3_finalMode_ = True
                                    d_4_breakAfterClose_ = True
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_14_next_]))
                                if (d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    generated = out10_
                                    insideConstrainedOut = out11_
                                    currentConstrainedOut = out12_
                                    d_2_spanEverOpened_ = True
                                    if (d_1_steps_) >= ((d_7_forceOpenAt_) - (30)):
                                        d_3_finalMode_ = True
                                        d_4_breakAfterClose_ = True
                    elif True:
                        d_18_cg_: _dafny.Seq
                        d_19_ci_: bool
                        d_20_cc_: _dafny.Seq
                        d_21_closed_: bool
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: _dafny.Seq
                        out16_: bool
                        out13_, out14_, out15_, out16_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_18_cg_ = out13_
                        d_19_ci_ = out14_
                        d_20_cc_ = out15_
                        d_21_closed_ = out16_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_21_closed_:
                            generated = d_18_cg_
                            insideConstrainedOut = d_19_ci_
                            currentConstrainedOut = d_20_cc_
                            if (d_4_breakAfterClose_) or (d_3_finalMode_):
                                raise _dafny.Break("0")
                        elif True:
                            d_22_constrainedPrompt_: _dafny.Seq
                            d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_23_next_: _dafny.Seq
                            out17_: _dafny.Seq
                            out17_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_23_next_ = out17_
                            if (d_23_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_24_ag_: _dafny.Seq
                                d_25_ai_: bool
                                d_26_ac_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_next_)
                                d_24_ag_ = out18_
                                d_25_ai_ = out19_
                                d_26_ac_ = out20_
                                generated = d_24_ag_
                                insideConstrainedOut = d_25_ai_
                                currentConstrainedOut = d_26_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_27_closeBudget_: int
            d_27_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_28_cg_: _dafny.Seq
            d_29_ci_: bool
            d_30_cc_: _dafny.Seq
            out21_: _dafny.Seq
            out22_: bool
            out23_: _dafny.Seq
            out21_, out22_, out23_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_27_closeBudget_)
            d_28_cg_ = out21_
            d_29_ci_ = out22_
            d_30_cc_ = out23_
            generated = d_28_cg_
            insideConstrainedOut = d_29_ci_
            currentConstrainedOut = d_30_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

