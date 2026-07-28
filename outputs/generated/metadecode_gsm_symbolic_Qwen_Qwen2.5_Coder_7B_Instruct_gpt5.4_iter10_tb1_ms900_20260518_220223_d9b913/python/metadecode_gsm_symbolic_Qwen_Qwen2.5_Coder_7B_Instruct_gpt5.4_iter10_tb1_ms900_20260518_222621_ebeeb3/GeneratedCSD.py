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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Put every arithmetic computation in a visible << >> span, and keep each span to only the computation.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_forcedOpenUsed_: bool
        d_2_forcedOpenUsed_ = insideConstrained
        d_3_outsideTokens_: int
        d_3_outsideTokens_ = 0
        d_4_preludeLimit_: int
        d_4_preludeLimit_ = 12
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_2_forcedOpenUsed_)) and ((d_3_outsideTokens_) >= (d_4_preludeLimit_)):
                            d_5_openedGenerated_: _dafny.Seq
                            d_6_openedInside_: bool
                            d_7_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_openedGenerated_ = out0_
                            d_6_openedInside_ = out1_
                            d_7_openedCurrent_ = out2_
                            generated = d_5_openedGenerated_
                            insideConstrainedOut = d_6_openedInside_
                            currentConstrainedOut = d_7_openedCurrent_
                            d_2_forcedOpenUsed_ = True
                            d_3_outsideTokens_ = 0
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                d_3_outsideTokens_ = (d_3_outsideTokens_) + (1)
                                if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_9_enteredGenerated_: _dafny.Seq
                                    d_10_enteredInside_: bool
                                    d_11_enteredCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_9_enteredGenerated_ = out4_
                                    d_10_enteredInside_ = out5_
                                    d_11_enteredCurrent_ = out6_
                                    generated = d_9_enteredGenerated_
                                    insideConstrainedOut = d_10_enteredInside_
                                    currentConstrainedOut = d_11_enteredCurrent_
                                    d_2_forcedOpenUsed_ = True
                                    d_3_outsideTokens_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_12_closedGenerated_: _dafny.Seq
                        d_13_closedInside_: bool
                        d_14_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_12_closedGenerated_ = out7_
                        d_13_closedInside_ = out8_
                        d_14_closedCurrent_ = out9_
                        generated = d_12_closedGenerated_
                        insideConstrainedOut = d_13_closedInside_
                        currentConstrainedOut = d_14_closedCurrent_
                        d_3_outsideTokens_ = 0
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_15_stablePrefix_: _dafny.Seq
                        d_15_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_16_constrainedPrompt_: _dafny.Seq
                        d_16_constrainedPrompt_ = (prompt) + (d_15_stablePrefix_)
                        d_17_remaining_: int
                        d_17_remaining_ = (maxSteps) - (d_1_steps_)
                        d_18_symbolBudget_: int
                        if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_17_remaining_)):
                            d_18_symbolBudget_ = d_17_remaining_
                        elif True:
                            d_18_symbolBudget_ = stepTokenBudget
                        d_19_symbolGenerated_: _dafny.Seq
                        d_20_symbolOut_: _dafny.Seq
                        d_21_hitEos_: bool
                        d_22_stepsUsed_: int
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: int
                        out10_, out11_, out12_, out13_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_16_constrainedPrompt_, generated, currentConstrainedOut, d_18_symbolBudget_, eosToken)
                        d_19_symbolGenerated_ = out10_
                        d_20_symbolOut_ = out11_
                        d_21_hitEos_ = out12_
                        d_22_stepsUsed_ = out13_
                        generated = d_19_symbolGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_20_symbolOut_
                        d_1_steps_ = (d_1_steps_) + (d_22_stepsUsed_)
                        if d_21_hitEos_:
                            d_23_repairedGenerated_: _dafny.Seq
                            d_24_repairedCurrent_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: _dafny.Seq
                            out14_, out15_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_23_repairedGenerated_ = out14_
                            d_24_repairedCurrent_ = out15_
                            generated = d_23_repairedGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_24_repairedCurrent_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_25_closedGenerated2_: _dafny.Seq
                                d_26_closedInside2_: bool
                                d_27_closedCurrent2_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_25_closedGenerated2_ = out16_
                                d_26_closedInside2_ = out17_
                                d_27_closedCurrent2_ = out18_
                                generated = d_25_closedGenerated2_
                                insideConstrainedOut = d_26_closedInside2_
                                currentConstrainedOut = d_27_closedCurrent2_
                                d_3_outsideTokens_ = 0
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

