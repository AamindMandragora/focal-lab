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
                    elif True:
                        d_8_stablePrefix_: _dafny.Seq
                        d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_9_constrainedPrompt_: _dafny.Seq
                        d_9_constrainedPrompt_ = (prompt) + (d_8_stablePrefix_)
                        d_10_validCount_: int
                        out6_: int
                        out6_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_10_validCount_ = out6_
                        d_11_deadEndSoon_: bool
                        out7_: bool
                        out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                        d_11_deadEndSoon_ = out7_
                        d_12_remaining_: int
                        d_12_remaining_ = (maxSteps) - (d_1_steps_)
                        if ((((stepTokenBudget) > (1)) and ((d_12_remaining_) > (0))) and ((d_10_validCount_) > (10))) and (not(d_11_deadEndSoon_)):
                            d_13_symbolBudget_: int
                            if (stepTokenBudget) > (d_12_remaining_):
                                d_13_symbolBudget_ = d_12_remaining_
                            elif True:
                                d_13_symbolBudget_ = stepTokenBudget
                            d_14_symbolGenerated_: _dafny.Seq
                            d_15_symbolOut_: _dafny.Seq
                            d_16_hitEos_: bool
                            d_17_stepsUsed_: int
                            out8_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: int
                            out8_, out9_, out10_, out11_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_9_constrainedPrompt_, generated, currentConstrainedOut, d_13_symbolBudget_, eosToken)
                            d_14_symbolGenerated_ = out8_
                            d_15_symbolOut_ = out9_
                            d_16_hitEos_ = out10_
                            d_17_stepsUsed_ = out11_
                            generated = d_14_symbolGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_15_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_17_stepsUsed_)
                            if d_16_hitEos_:
                                raise _dafny.Break("0")
                        elif (d_11_deadEndSoon_) or ((d_10_validCount_) <= (4)):
                            d_18_nextHard_: _dafny.Seq
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_18_nextHard_ = out12_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_18_nextHard_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_19_appendedGenerated1_: _dafny.Seq
                                d_20_appendedInside1_: bool
                                d_21_appendedCurrent1_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_nextHard_)
                                d_19_appendedGenerated1_ = out13_
                                d_20_appendedInside1_ = out14_
                                d_21_appendedCurrent1_ = out15_
                                generated = d_19_appendedGenerated1_
                                insideConstrainedOut = d_20_appendedInside1_
                                currentConstrainedOut = d_21_appendedCurrent1_
                        elif True:
                            d_22_nextGroup_: _dafny.Seq
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                            d_22_nextGroup_ = out16_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_22_nextGroup_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_23_appendedGenerated2_: _dafny.Seq
                                d_24_appendedInside2_: bool
                                d_25_appendedCurrent2_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_nextGroup_)
                                d_23_appendedGenerated2_ = out17_
                                d_24_appendedInside2_ = out18_
                                d_25_appendedCurrent2_ = out19_
                                generated = d_23_appendedGenerated2_
                                insideConstrainedOut = d_24_appendedInside2_
                                currentConstrainedOut = d_25_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

