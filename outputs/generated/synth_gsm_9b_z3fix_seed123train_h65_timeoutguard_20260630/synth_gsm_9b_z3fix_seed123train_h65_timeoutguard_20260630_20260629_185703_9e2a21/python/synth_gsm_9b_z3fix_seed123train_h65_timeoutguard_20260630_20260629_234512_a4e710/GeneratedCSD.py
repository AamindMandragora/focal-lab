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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. Use the variable names from the problem directly (no curly braces). Write your final answer as <<expression>> where the expression uses variable names, integers, and operators (+, -, *, /, //, %, int()). Example: if the answer involves n items at cost c each, write <<n * c>>. Write the final answer expression at the very end of your solution.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanSteps_: int
        d_2_spanSteps_ = 0
        d_3_spanBudget_: int
        d_3_spanBudget_ = 70
        d_4_earlyStopThreshold_: int
        d_4_earlyStopThreshold_ = 150
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_remaining_: int
                        d_5_remaining_ = (maxSteps) - (d_1_steps_)
                        if (d_5_remaining_) <= (2):
                            raise _dafny.Break("0")
                        d_6_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_6_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_6_next_) == (eosToken):
                            raise _dafny.Break("0")
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                        if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_2_spanSteps_ = 0
                    elif (d_2_spanSteps_) >= (d_3_spanBudget_):
                        d_7_remaining_: int
                        d_7_remaining_ = (maxSteps) - (d_1_steps_)
                        if (d_7_remaining_) == (0):
                            raise _dafny.Break("0")
                        d_8_closeBudget2_: int
                        if (d_7_remaining_) < (30):
                            d_8_closeBudget2_ = d_7_remaining_
                        elif True:
                            d_8_closeBudget2_ = 30
                        d_9_cg2_: _dafny.Seq
                        d_10_ci2_: bool
                        d_11_cc2_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_8_closeBudget2_)
                        d_9_cg2_ = out1_
                        d_10_ci2_ = out2_
                        d_11_cc2_ = out3_
                        generated = d_9_cg2_
                        insideConstrainedOut = d_10_ci2_
                        currentConstrainedOut = d_11_cc2_
                        d_1_steps_ = (d_1_steps_) + (d_8_closeBudget2_)
                        d_2_spanSteps_ = 0
                        if (d_1_steps_) > (d_4_earlyStopThreshold_):
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_12_remaining_: int
                        d_12_remaining_ = (maxSteps) - (d_1_steps_)
                        if (d_12_remaining_) == (0):
                            raise _dafny.Break("0")
                        d_13_closeBudget_: int
                        if (d_12_remaining_) < (20):
                            d_13_closeBudget_ = d_12_remaining_
                        elif True:
                            d_13_closeBudget_ = 20
                        d_14_cg_: _dafny.Seq
                        d_15_ci_: bool
                        d_16_cc_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_13_closeBudget_)
                        d_14_cg_ = out4_
                        d_15_ci_ = out5_
                        d_16_cc_ = out6_
                        generated = d_14_cg_
                        insideConstrainedOut = d_15_ci_
                        currentConstrainedOut = d_16_cc_
                        d_1_steps_ = (d_1_steps_) + (d_13_closeBudget_)
                        d_2_spanSteps_ = 0
                        if (d_1_steps_) > (d_4_earlyStopThreshold_):
                            raise _dafny.Break("0")
                    elif True:
                        d_17_constrainedPrompt_: _dafny.Seq
                        d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_18_next_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                        d_18_next_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanSteps_ = (d_2_spanSteps_) + (1)
                        if (d_18_next_) == (eosToken):
                            d_19_remaining_: int
                            d_19_remaining_ = (maxSteps) - (d_1_steps_)
                            if (d_19_remaining_) == (0):
                                raise _dafny.Break("0")
                            d_20_closeBudget3_: int
                            if (d_19_remaining_) < (25):
                                d_20_closeBudget3_ = d_19_remaining_
                            elif True:
                                d_20_closeBudget3_ = 25
                            d_21_cg3_: _dafny.Seq
                            d_22_ci3_: bool
                            d_23_cc3_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_20_closeBudget3_)
                            d_21_cg3_ = out8_
                            d_22_ci3_ = out9_
                            d_23_cc3_ = out10_
                            generated = d_21_cg3_
                            insideConstrainedOut = d_22_ci3_
                            currentConstrainedOut = d_23_cc3_
                            d_1_steps_ = (d_1_steps_) + (d_20_closeBudget3_)
                            raise _dafny.Break("0")
                        elif True:
                            d_24_ag_: _dafny.Seq
                            d_25_ai_: bool
                            d_26_ac_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                            d_24_ag_ = out11_
                            d_25_ai_ = out12_
                            d_26_ac_ = out13_
                            generated = d_24_ag_
                            insideConstrainedOut = d_25_ai_
                            currentConstrainedOut = d_26_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

