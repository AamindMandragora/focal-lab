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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem symbolically. The decoder may open the first visible << >> span immediately, so make the first span contain the final answer expression, not a scratch placeholder. Inside any span write only a valid arithmetic expression: no prose, no units, no Markdown, no LaTeX. Preserve variable names exactly as given, including underscores, and use all relevant quantities from the problem. Do not invent generic variables such as t, tf, total, or remaining unless those exact variables appear in the problem. Do not multiply by currency marker variables such as cur. Prefer direct formulas: total cost is sums of quantity*unit_price plus fixed costs; fraction-of-fraction counts are int(n*frac_1*frac_2); days lasting is n//(daily_amounts); percent increase price is price+price*percent/100.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_openedGenerated_: _dafny.Seq
                        d_3_openedInside_: bool
                        d_4_openedCurrent_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_2_openedGenerated_ = out0_
                        d_3_openedInside_ = out1_
                        d_4_openedCurrent_ = out2_
                        generated = d_2_openedGenerated_
                        insideConstrainedOut = d_3_openedInside_
                        currentConstrainedOut = d_4_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_5_closedGenerated_: _dafny.Seq
                        d_6_closedInside_: bool
                        d_7_closedCurrent_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_5_closedGenerated_ = out3_
                        d_6_closedInside_ = out4_
                        d_7_closedCurrent_ = out5_
                        generated = d_5_closedGenerated_
                        insideConstrainedOut = d_6_closedInside_
                        currentConstrainedOut = d_7_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_8_stablePrefix_: _dafny.Seq
                        d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_9_constrainedPrompt_: _dafny.Seq
                        d_9_constrainedPrompt_ = (prompt) + (d_8_stablePrefix_)
                        d_10_symbolBudget_: int
                        d_10_symbolBudget_ = (maxSteps) - (d_1_steps_)
                        d_11_symbolGenerated_: _dafny.Seq
                        d_12_symbolOut_: _dafny.Seq
                        d_13_hitEos_: bool
                        d_14_stepsUsed_: int
                        out6_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: int
                        out6_, out7_, out8_, out9_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_9_constrainedPrompt_, generated, currentConstrainedOut, d_10_symbolBudget_, eosToken)
                        d_11_symbolGenerated_ = out6_
                        d_12_symbolOut_ = out7_
                        d_13_hitEos_ = out8_
                        d_14_stepsUsed_ = out9_
                        generated = d_11_symbolGenerated_
                        currentConstrainedOut = d_12_symbolOut_
                        d_1_steps_ = (d_1_steps_) + (d_14_stepsUsed_)
                        if d_13_hitEos_:
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

