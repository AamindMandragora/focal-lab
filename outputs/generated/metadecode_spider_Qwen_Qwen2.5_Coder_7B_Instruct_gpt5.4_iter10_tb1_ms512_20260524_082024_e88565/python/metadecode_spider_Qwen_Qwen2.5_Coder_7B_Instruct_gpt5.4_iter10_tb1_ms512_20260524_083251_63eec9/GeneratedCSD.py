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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one SQL query in a single visible constrained span and no explanation.")))
        (d_0_helpers_).SetNonDeterministic(lm, False)
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openCount_: int = int(0)
        out0_: int
        out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generatedPrefix, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        d_2_openCount_ = out0_
        d_3_openedAnySpan_: bool
        d_3_openedAnySpan_ = insideConstrained
        if not(d_3_openedAnySpan_):
            d_3_openedAnySpan_ = (d_2_openCount_) > (0)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_3_openedAnySpan_:
                            d_4_nextFree_: _dafny.Seq
                            out1_: _dafny.Seq
                            out1_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_4_nextFree_ = out1_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_4_nextFree_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_nextFree_]))
                                if (d_4_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                elif True:
                                    raise _dafny.Break("0")
                        elif True:
                            d_5_openedGenerated_: _dafny.Seq
                            d_6_openedInside_: bool
                            d_7_openedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_openedGenerated_ = out2_
                            d_6_openedInside_ = out3_
                            d_7_openedCurrent_ = out4_
                            generated = d_5_openedGenerated_
                            insideConstrainedOut = d_6_openedInside_
                            currentConstrainedOut = d_7_openedCurrent_
                            d_3_openedAnySpan_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_8_closedGenerated_: _dafny.Seq
                        d_9_closedInside_: bool
                        d_10_closedCurrent_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_8_closedGenerated_ = out5_
                        d_9_closedInside_ = out6_
                        d_10_closedCurrent_ = out7_
                        generated = d_8_closedGenerated_
                        insideConstrainedOut = d_9_closedInside_
                        currentConstrainedOut = d_10_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_11_stablePrefix_: _dafny.Seq
                        d_11_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_12_constrainedPrompt_: _dafny.Seq
                        d_12_constrainedPrompt_ = (prompt) + (d_11_stablePrefix_)
                        d_13_remaining_: int
                        d_13_remaining_ = (maxSteps) - (d_1_steps_)
                        d_14_symbolBudget_: int
                        d_14_symbolBudget_ = stepTokenBudget
                        if ((d_14_symbolBudget_) == (0)) or ((d_14_symbolBudget_) > (d_13_remaining_)):
                            d_14_symbolBudget_ = d_13_remaining_
                        d_15_symbolGenerated_: _dafny.Seq
                        d_16_symbolCurrent_: _dafny.Seq
                        d_17_hitEos_: bool
                        d_18_stepsUsed_: int
                        out8_: _dafny.Seq
                        out9_: _dafny.Seq
                        out10_: bool
                        out11_: int
                        out8_, out9_, out10_, out11_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_12_constrainedPrompt_, generated, currentConstrainedOut, d_14_symbolBudget_, eosToken)
                        d_15_symbolGenerated_ = out8_
                        d_16_symbolCurrent_ = out9_
                        d_17_hitEos_ = out10_
                        d_18_stepsUsed_ = out11_
                        generated = d_15_symbolGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_16_symbolCurrent_
                        d_1_steps_ = (d_1_steps_) + (d_18_stepsUsed_)
                        if d_17_hitEos_:
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

