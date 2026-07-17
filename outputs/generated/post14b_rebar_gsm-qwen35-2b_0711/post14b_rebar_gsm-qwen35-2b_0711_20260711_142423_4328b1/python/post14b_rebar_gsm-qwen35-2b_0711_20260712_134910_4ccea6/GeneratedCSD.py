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
        if (maxSteps) == (0):
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step using the given variable names. At the end, write: The final answer is <<EXPRESSION>> where EXPRESSION is a single Python arithmetic expression. Use bare variable names without braces or dollar signs. Use only +, -, *, /, //, %, int(), and parentheses. Write the complete expression (not just one variable). Examples: <<n1 * p1 + n2 * p2>>, <<int(n * frac)>>, <<(a + b) * c // d>>, <<total + n2 - n1>>, <<count * (n1 + n2 + n3 + n4 + n5)>>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_freeBudget_: int
        if (maxSteps) >= (150):
            d_3_freeBudget_ = 150
        elif True:
            d_3_freeBudget_ = maxSteps
        while ((d_2_steps_) < (d_3_freeBudget_)) and (not(insideConstrainedOut)):
            d_4_next_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_4_next_ = out0_
            d_2_steps_ = (d_2_steps_) + (1)
            if (d_4_next_) == (eosToken):
                cost = d_2_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                insideConstrainedOut = True
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_5_og_: _dafny.Seq
            d_6_oi_: bool
            d_7_oc_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_5_og_ = out1_
            d_6_oi_ = out2_
            d_7_oc_ = out3_
            generated = d_5_og_
            insideConstrainedOut = d_6_oi_
            currentConstrainedOut = d_7_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_8_remaining_: int
            d_8_remaining_ = (maxSteps) - (d_2_steps_)
            d_9_symbolBudget_: int
            if (d_8_remaining_) >= (60):
                d_9_symbolBudget_ = 60
            elif True:
                d_9_symbolBudget_ = d_8_remaining_
            if (d_9_symbolBudget_) > (0):
                d_10_constrainedPrompt_: _dafny.Seq
                d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_11_gOut_: _dafny.Seq
                d_12_cOut_: _dafny.Seq
                d_13_hitEos_: bool
                d_14_stepsUsed_: int
                out4_: _dafny.Seq
                out5_: _dafny.Seq
                out6_: bool
                out7_: int
                out4_, out5_, out6_, out7_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_10_constrainedPrompt_, generated, currentConstrainedOut, d_9_symbolBudget_, eosToken)
                d_11_gOut_ = out4_
                d_12_cOut_ = out5_
                d_13_hitEos_ = out6_
                d_14_stepsUsed_ = out7_
                generated = d_11_gOut_
                currentConstrainedOut = d_12_cOut_
                d_2_steps_ = (d_2_steps_) + (d_14_stepsUsed_)
                if d_13_hitEos_:
                    if (d_2_steps_) < (maxSteps):
                        d_15_remaining2_: int
                        d_15_remaining2_ = (maxSteps) - (d_2_steps_)
                        d_16_closeBudget_: int
                        if (d_15_remaining2_) >= (5):
                            d_16_closeBudget_ = 5
                        elif True:
                            d_16_closeBudget_ = d_15_remaining2_
                        if (d_16_closeBudget_) > (0):
                            d_17_cgOut_: _dafny.Seq
                            d_18_ciOut_: bool
                            d_19_ccOut_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_16_closeBudget_)
                            d_17_cgOut_ = out8_
                            d_18_ciOut_ = out9_
                            d_19_ccOut_ = out10_
                            generated = d_17_cgOut_
                            insideConstrainedOut = d_18_ciOut_
                            currentConstrainedOut = d_19_ccOut_
                            d_2_steps_ = (d_2_steps_) + (d_16_closeBudget_)
                    cost = d_2_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_20_remaining_: int
            d_20_remaining_ = (maxSteps) - (d_2_steps_)
            d_21_closeBudget_: int
            if (d_20_remaining_) >= (10):
                d_21_closeBudget_ = 10
            elif True:
                d_21_closeBudget_ = d_20_remaining_
            if (d_21_closeBudget_) > (0):
                d_22_cgOut_: _dafny.Seq
                d_23_ciOut_: bool
                d_24_ccOut_: _dafny.Seq
                out11_: _dafny.Seq
                out12_: bool
                out13_: _dafny.Seq
                out11_, out12_, out13_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_21_closeBudget_)
                d_22_cgOut_ = out11_
                d_23_ciOut_ = out12_
                d_24_ccOut_ = out13_
                generated = d_22_cgOut_
                insideConstrainedOut = d_23_ciOut_
                currentConstrainedOut = d_24_ccOut_
                d_2_steps_ = (d_2_steps_) + (d_21_closeBudget_)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

