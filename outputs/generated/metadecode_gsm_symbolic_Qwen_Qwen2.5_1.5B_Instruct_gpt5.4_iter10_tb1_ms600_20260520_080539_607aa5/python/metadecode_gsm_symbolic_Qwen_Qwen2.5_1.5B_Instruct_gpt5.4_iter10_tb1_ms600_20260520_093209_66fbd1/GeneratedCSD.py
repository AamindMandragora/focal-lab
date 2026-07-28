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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Put intermediate symbolic expressions and the final answer inside visible << >> delimiters. Keep each delimited expression syntactically valid.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openCount_: int
        if insideConstrained:
            d_2_openCount_ = 1
        elif True:
            d_2_openCount_ = 0
        d_3_rollbackLimit_: int
        d_3_rollbackLimit_ = 24
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_2_openCount_) == (0):
                            d_4_openedGenerated_: _dafny.Seq
                            d_5_openedInside_: bool
                            d_6_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_openedGenerated_ = out0_
                            d_5_openedInside_ = out1_
                            d_6_openedCurrent_ = out2_
                            generated = d_4_openedGenerated_
                            insideConstrainedOut = d_5_openedInside_
                            currentConstrainedOut = d_6_openedCurrent_
                            d_2_openCount_ = 1
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_7_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_8_closedGenerated_: _dafny.Seq
                        d_9_closedInside_: bool
                        d_10_closedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_8_closedGenerated_ = out4_
                        d_9_closedInside_ = out5_
                        d_10_closedCurrent_ = out6_
                        generated = d_8_closedGenerated_
                        insideConstrainedOut = d_9_closedInside_
                        currentConstrainedOut = d_10_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif (len(currentConstrainedOut)) >= (d_3_rollbackLimit_):
                        d_11_rolledGenerated_: _dafny.Seq
                        d_12_rolledCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: _dafny.Seq
                        out7_, out8_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_11_rolledGenerated_ = out7_
                        d_12_rolledCurrent_ = out8_
                        generated = d_11_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_12_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_13_stablePrefix_: _dafny.Seq
                        d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_14_constrainedPrompt_: _dafny.Seq
                        d_14_constrainedPrompt_ = (prompt) + (d_13_stablePrefix_)
                        d_15_remaining_: int
                        d_15_remaining_ = (maxSteps) - (d_1_steps_)
                        d_16_symbolBudget_: int
                        if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_15_remaining_)):
                            d_16_symbolBudget_ = d_15_remaining_
                        elif True:
                            d_16_symbolBudget_ = stepTokenBudget
                        d_17_symbolGenerated_: _dafny.Seq
                        d_18_symbolOut_: _dafny.Seq
                        d_19_hitEos_: bool
                        d_20_stepsUsed_: int
                        out9_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: int
                        out9_, out10_, out11_, out12_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_14_constrainedPrompt_, generated, currentConstrainedOut, d_16_symbolBudget_, eosToken)
                        d_17_symbolGenerated_ = out9_
                        d_18_symbolOut_ = out10_
                        d_19_hitEos_ = out11_
                        d_20_stepsUsed_ = out12_
                        generated = d_17_symbolGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_18_symbolOut_
                        d_1_steps_ = (d_1_steps_) + (d_20_stepsUsed_)
                        if d_19_hitEos_:
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

