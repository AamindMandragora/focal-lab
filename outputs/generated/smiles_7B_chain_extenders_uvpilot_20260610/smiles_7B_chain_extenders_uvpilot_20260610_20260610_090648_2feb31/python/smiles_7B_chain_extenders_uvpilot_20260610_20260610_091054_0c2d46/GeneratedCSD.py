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
        d_1_steps_: int
        d_1_steps_ = 0
        if not(insideConstrained):
            d_2_budget_: int
            d_2_budget_ = maxSteps
            if (d_2_budget_) > (0):
                d_3_constrainedOut_: _dafny.Seq
                d_4_terminatedByEos_: bool
                out0_: _dafny.Seq
                out1_: bool
                out0_, out1_ = (d_0_helpers_).ConstrainedGeneration(lm, parser, prompt, d_2_budget_, eosToken)
                d_3_constrainedOut_ = out0_
                d_4_terminatedByEos_ = out1_
                generated = (generatedPrefix) + (d_3_constrainedOut_)
                d_1_steps_ = d_2_budget_
            cost = d_1_steps_
        elif True:
            with _dafny.label("1_0"):
                while (d_1_steps_) < (maxSteps):
                    with _dafny.c_label("1_0"):
                        if not(insideConstrainedOut):
                            raise _dafny.Break("1_0")
                        elif (parser).IsCompletePrefix(currentConstrainedOut):
                            if (d_1_steps_) < (maxSteps):
                                d_5_cg_: _dafny.Seq
                                d_6_ci_: bool
                                d_7_cc_: _dafny.Seq
                                out2_: _dafny.Seq
                                out3_: bool
                                out4_: _dafny.Seq
                                out2_, out3_, out4_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_5_cg_ = out2_
                                d_6_ci_ = out3_
                                d_7_cc_ = out4_
                                generated = d_5_cg_
                                insideConstrainedOut = d_6_ci_
                                currentConstrainedOut = d_7_cc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("1_0")
                        elif True:
                            d_8_constrainedPrompt_: _dafny.Seq
                            d_8_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_9_next_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_9_next_ = out5_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_10_cg_: _dafny.Seq
                                    d_11_ci_: bool
                                    d_12_cc_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out7_: bool
                                    out8_: _dafny.Seq
                                    out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_10_cg_ = out6_
                                    d_11_ci_ = out7_
                                    d_12_cc_ = out8_
                                    generated = d_10_cg_
                                    insideConstrainedOut = d_11_ci_
                                    currentConstrainedOut = d_12_cc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("1_0")
                            elif True:
                                d_13_ag_: _dafny.Seq
                                d_14_ai_: bool
                                d_15_ac_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                                d_13_ag_ = out9_
                                d_14_ai_ = out10_
                                d_15_ac_ = out11_
                                generated = d_13_ag_
                                insideConstrainedOut = d_14_ai_
                                currentConstrainedOut = d_15_ac_
                        pass
                pass
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

