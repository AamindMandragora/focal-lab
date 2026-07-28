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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. Show all reasoning. Place the FINAL symbolic answer inside << >>. Use ONLY variable names from the problem (NO {curly} braces). Allowed operators: +, -, *, /, //, %, (, ), int(). Use int() for integer results. Examples: <<int(n * price)>> or <<n - (n1*w1 + n2*w2)>> or <<int((length + space) / (width + space))>>. Keep expression minimal and direct.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanEverOpened_: bool
        d_2_spanEverOpened_ = insideConstrained
        d_3_minConstrainedBudget_: int
        d_3_minConstrainedBudget_ = 80
        d_4_fracFree_: int
        d_4_fracFree_ = _dafny.euclidian_division((maxSteps) * (65), 100)
        d_5_freeLimit_: int = int(0)
        if ((d_4_fracFree_) + (d_3_minConstrainedBudget_)) <= (maxSteps):
            d_5_freeLimit_ = d_4_fracFree_
        elif (d_3_minConstrainedBudget_) <= (maxSteps):
            d_5_freeLimit_ = (maxSteps) - (d_3_minConstrainedBudget_)
        elif True:
            d_5_freeLimit_ = 0
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    if (not(d_2_spanEverOpened_)) and ((d_1_steps_) >= (d_5_freeLimit_)):
                        d_6_og_: _dafny.Seq
                        d_7_oi_: bool
                        d_8_oc_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_6_og_ = out0_
                        d_7_oi_ = out1_
                        d_8_oc_ = out2_
                        generated = d_6_og_
                        insideConstrainedOut = d_7_oi_
                        currentConstrainedOut = d_8_oc_
                        d_2_spanEverOpened_ = True
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_9_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_9_next_ = out3_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_9_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                            if (d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                generated = out4_
                                insideConstrainedOut = out5_
                                currentConstrainedOut = out6_
                                d_2_spanEverOpened_ = True
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_10_remBudget_: int
            d_10_remBudget_ = (maxSteps) - (d_1_steps_)
            d_11_groundBudget_: int
            d_11_groundBudget_ = _dafny.euclidian_division(d_10_remBudget_, 2)
            if (d_11_groundBudget_) < (1):
                d_11_groundBudget_ = 1
            d_12_closeBudget_: int
            d_12_closeBudget_ = (d_10_remBudget_) - (d_11_groundBudget_)
            if ((d_12_closeBudget_) < (1)) and ((d_10_remBudget_) >= (2)):
                d_12_closeBudget_ = 1
                d_11_groundBudget_ = (d_10_remBudget_) - (1)
            if (d_12_closeBudget_) < (1):
                d_12_closeBudget_ = d_10_remBudget_
                d_11_groundBudget_ = 0
            if (d_11_groundBudget_) >= (1):
                d_13_stable_: _dafny.Seq
                d_13_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_14_constrainedPrompt_: _dafny.Seq
                d_14_constrainedPrompt_ = (prompt) + (d_13_stable_)
                d_15_filled_: _dafny.Seq
                out7_: _dafny.Seq
                out7_ = (d_0_helpers_).RegenerateUnitOnGroundingFailure(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, eosToken, d_11_groundBudget_, 3, d_11_groundBudget_)
                d_15_filled_ = out7_
                generated = (d_13_stable_) + (d_15_filled_)
                currentConstrainedOut = d_15_filled_
                d_1_steps_ = (d_1_steps_) + (d_11_groundBudget_)
            if (d_1_steps_) < (maxSteps):
                d_16_closeBudget2_: int
                d_16_closeBudget2_ = (maxSteps) - (d_1_steps_)
                d_17_cg_: _dafny.Seq
                d_18_ci_: bool
                d_19_cc_: _dafny.Seq
                out8_: _dafny.Seq
                out9_: bool
                out10_: _dafny.Seq
                out8_, out9_, out10_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_16_closeBudget2_)
                d_17_cg_ = out8_
                d_18_ci_ = out9_
                d_19_cc_ = out10_
                generated = d_17_cg_
                insideConstrainedOut = d_18_ci_
                currentConstrainedOut = d_19_cc_
                d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

