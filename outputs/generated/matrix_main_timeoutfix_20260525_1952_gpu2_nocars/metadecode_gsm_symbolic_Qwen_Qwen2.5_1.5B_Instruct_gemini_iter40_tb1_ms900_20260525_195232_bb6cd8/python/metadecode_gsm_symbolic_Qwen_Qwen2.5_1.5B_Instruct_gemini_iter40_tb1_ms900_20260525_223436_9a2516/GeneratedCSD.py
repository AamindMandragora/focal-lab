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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem by building a symbolic expression. Carefully map the problem's text to variables and operations. Double-check your final expression for correctness.")))
        d_1_narrowThreshold_: int
        d_1_narrowThreshold_ = 10
        d_2_rollbackLimit_: int
        d_2_rollbackLimit_ = 30
        d_3_penaltyTokens_: _dafny.Seq
        d_3_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/"))])
        d_4_steps_: int
        d_4_steps_ = 0
        with _dafny.label("0"):
            while (d_4_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_5_next_ = out0_
                        d_4_steps_ = (d_4_steps_) + (1)
                        if (d_5_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                            if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_6_closedGenerated_: _dafny.Seq
                            d_7_closedInside_: bool
                            d_8_closedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_6_closedGenerated_ = out1_
                            d_7_closedInside_ = out2_
                            d_8_closedCurrent_ = out3_
                            generated = d_6_closedGenerated_
                            insideConstrainedOut = d_7_closedInside_
                            currentConstrainedOut = d_8_closedCurrent_
                            d_4_steps_ = (d_4_steps_) + (1)
                        elif (len(currentConstrainedOut)) >= (d_2_rollbackLimit_):
                            d_9_rolledGenerated_: _dafny.Seq
                            d_10_rolledCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: _dafny.Seq
                            out4_, out5_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_9_rolledGenerated_ = out4_
                            d_10_rolledCurrent_ = out5_
                            generated = d_9_rolledGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_10_rolledCurrent_
                            d_4_steps_ = (d_4_steps_) + (1)
                        elif True:
                            d_11_constrainedPrompt_: _dafny.Seq
                            d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_12_validCount_: int
                            out6_: int
                            out6_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_12_validCount_ = out6_
                            if (d_12_validCount_) <= (d_1_narrowThreshold_):
                                d_13_next_: _dafny.Seq
                                out7_: _dafny.Seq
                                out7_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_penaltyTokens_, _dafny.BigRational('4e0'), d_1_narrowThreshold_, eosToken)
                                d_13_next_ = out7_
                                d_4_steps_ = (d_4_steps_) + (1)
                                if (d_13_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_14_appendedGenerated_: _dafny.Seq
                                    d_15_appendedInside_: bool
                                    d_16_appendedCurrent_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out9_: bool
                                    out10_: _dafny.Seq
                                    out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                                    d_14_appendedGenerated_ = out8_
                                    d_15_appendedInside_ = out9_
                                    d_16_appendedCurrent_ = out10_
                                    generated = d_14_appendedGenerated_
                                    insideConstrainedOut = d_15_appendedInside_
                                    currentConstrainedOut = d_16_appendedCurrent_
                            elif True:
                                d_17_remaining_: int
                                d_17_remaining_ = (maxSteps) - (d_4_steps_)
                                d_18_symbolBudget_: int
                                if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_17_remaining_)):
                                    d_18_symbolBudget_ = d_17_remaining_
                                elif True:
                                    d_18_symbolBudget_ = stepTokenBudget
                                d_19_symbolGenerated_: _dafny.Seq
                                d_20_symbolOut_: _dafny.Seq
                                d_21_hitEos_: bool
                                d_22_stepsUsed_: int
                                out11_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: int
                                out11_, out12_, out13_, out14_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_11_constrainedPrompt_, generated, currentConstrainedOut, d_18_symbolBudget_, eosToken)
                                d_19_symbolGenerated_ = out11_
                                d_20_symbolOut_ = out12_
                                d_21_hitEos_ = out13_
                                d_22_stepsUsed_ = out14_
                                generated = d_19_symbolGenerated_
                                currentConstrainedOut = d_20_symbolOut_
                                d_4_steps_ = (d_4_steps_) + (d_22_stepsUsed_)
                                if d_21_hitEos_:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_4_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

