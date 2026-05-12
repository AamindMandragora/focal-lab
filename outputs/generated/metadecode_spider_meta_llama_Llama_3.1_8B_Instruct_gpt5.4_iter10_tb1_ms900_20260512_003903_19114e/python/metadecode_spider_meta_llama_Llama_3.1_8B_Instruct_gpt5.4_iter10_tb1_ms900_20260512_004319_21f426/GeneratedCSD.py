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
        d_2_openedHere_: bool
        d_2_openedHere_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_openedGenerated_: _dafny.Seq
                        d_4_openedInside_: bool
                        d_5_openedCurrent_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_3_openedGenerated_ = out0_
                        d_4_openedInside_ = out1_
                        d_5_openedCurrent_ = out2_
                        generated = d_3_openedGenerated_
                        insideConstrainedOut = d_4_openedInside_
                        currentConstrainedOut = d_5_openedCurrent_
                        d_2_openedHere_ = True
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_6_closedGenerated_: _dafny.Seq
                        d_7_closedInside_: bool
                        d_8_closedCurrent_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_6_closedGenerated_ = out3_
                        d_7_closedInside_ = out4_
                        d_8_closedCurrent_ = out5_
                        generated = d_6_closedGenerated_
                        insideConstrainedOut = d_7_closedInside_
                        currentConstrainedOut = d_8_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_9_remaining_: int
                        d_9_remaining_ = (maxSteps) - (d_1_steps_)
                        d_10_stablePrefix_: _dafny.Seq
                        d_10_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_11_constrainedPrompt_: _dafny.Seq
                        d_11_constrainedPrompt_ = (prompt) + (d_10_stablePrefix_)
                        d_12_validCount_: int
                        out6_: int
                        out6_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_12_validCount_ = out6_
                        d_13_deadEndSoon_: bool
                        out7_: bool
                        out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                        d_13_deadEndSoon_ = out7_
                        if (d_9_remaining_) == (1):
                            if d_2_openedHere_:
                                d_14_rolledGenerated_: _dafny.Seq
                                d_15_rolledCurrent_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: _dafny.Seq
                                out8_, out9_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_14_rolledGenerated_ = out8_
                                d_15_rolledCurrent_ = out9_
                                generated = d_14_rolledGenerated_
                                insideConstrainedOut = True
                                currentConstrainedOut = d_15_rolledCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_16_lastTok_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_16_lastTok_ = out10_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_16_lastTok_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_17_appendedGenerated1_: _dafny.Seq
                                    d_18_appendedInside1_: bool
                                    d_19_appendedCurrent1_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_lastTok_)
                                    d_17_appendedGenerated1_ = out11_
                                    d_18_appendedInside1_ = out12_
                                    d_19_appendedCurrent1_ = out13_
                                    generated = d_17_appendedGenerated1_
                                    insideConstrainedOut = d_18_appendedInside1_
                                    currentConstrainedOut = d_19_appendedCurrent1_
                        elif ((((d_9_remaining_) > (1)) and ((stepTokenBudget) > (1))) and ((d_12_validCount_) > (6))) and (not(d_13_deadEndSoon_)):
                            d_20_usableBudget_: int
                            d_20_usableBudget_ = (d_9_remaining_) - (1)
                            d_21_symbolBudget_: int
                            if (stepTokenBudget) > (d_20_usableBudget_):
                                d_21_symbolBudget_ = d_20_usableBudget_
                            elif True:
                                d_21_symbolBudget_ = stepTokenBudget
                            d_22_symbolGenerated_: _dafny.Seq
                            d_23_symbolOut_: _dafny.Seq
                            d_24_hitEos_: bool
                            d_25_stepsUsed_: int
                            out14_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: int
                            out14_, out15_, out16_, out17_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_11_constrainedPrompt_, generated, currentConstrainedOut, d_21_symbolBudget_, eosToken)
                            d_22_symbolGenerated_ = out14_
                            d_23_symbolOut_ = out15_
                            d_24_hitEos_ = out16_
                            d_25_stepsUsed_ = out17_
                            generated = d_22_symbolGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_23_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_25_stepsUsed_)
                            if d_24_hitEos_:
                                raise _dafny.Break("0")
                        elif True:
                            d_26_next_: _dafny.Seq
                            d_26_next_ = eosToken
                            if ((d_12_validCount_) <= (3)) or (d_13_deadEndSoon_):
                                out18_: _dafny.Seq
                                out18_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_26_next_ = out18_
                            elif True:
                                d_27_gatedNext_: _dafny.Seq
                                d_28_wasConstrained_: bool
                                out19_: _dafny.Seq
                                out20_: bool
                                out19_, out20_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_27_gatedNext_ = out19_
                                d_28_wasConstrained_ = out20_
                                d_26_next_ = d_27_gatedNext_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_26_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_29_appendedGenerated2_: _dafny.Seq
                                d_30_appendedInside2_: bool
                                d_31_appendedCurrent2_: _dafny.Seq
                                out21_: _dafny.Seq
                                out22_: bool
                                out23_: _dafny.Seq
                                out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next_)
                                d_29_appendedGenerated2_ = out21_
                                d_30_appendedInside2_ = out22_
                                d_31_appendedCurrent2_ = out23_
                                generated = d_29_appendedGenerated2_
                                insideConstrainedOut = d_30_appendedInside2_
                                currentConstrainedOut = d_31_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

