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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Every arithmetic computation must appear inside visible << >> delimiters, and each << must be closed with >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_chunkCap_: int
        d_2_chunkCap_ = 24
        d_3_rollbackLimit_: int
        d_3_rollbackLimit_ = 24
        d_4_narrowThreshold_: int
        d_4_narrowThreshold_ = 12
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_remainingOut_: int
                        d_5_remainingOut_ = (maxSteps) - (d_1_steps_)
                        d_6_chunkBudget_: int
                        if (d_5_remainingOut_) < (d_2_chunkCap_):
                            d_6_chunkBudget_ = d_5_remainingOut_
                        elif True:
                            d_6_chunkBudget_ = d_2_chunkCap_
                        d_7_chunkedGenerated_: _dafny.Seq
                        d_8_stoppedOnOpenSpan_: bool
                        d_9_stoppedOnEos_: bool
                        d_10_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_7_chunkedGenerated_ = out0_
                        d_8_stoppedOnOpenSpan_ = out1_
                        d_9_stoppedOnEos_ = out2_
                        d_10_stepsUsed_ = out3_
                        generated = d_7_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_10_stepsUsed_)
                        if d_9_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_8_stoppedOnOpenSpan_:
                            d_11_enteredGenerated_: _dafny.Seq
                            d_12_enteredInside_: bool
                            d_13_enteredCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_11_enteredGenerated_ = out4_
                            d_12_enteredInside_ = out5_
                            d_13_enteredCurrent_ = out6_
                            generated = d_11_enteredGenerated_
                            insideConstrainedOut = d_12_enteredInside_
                            currentConstrainedOut = d_13_enteredCurrent_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_14_closedGenerated_: _dafny.Seq
                        d_15_closedInside_: bool
                        d_16_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_14_closedGenerated_ = out7_
                        d_15_closedInside_ = out8_
                        d_16_closedCurrent_ = out9_
                        generated = d_14_closedGenerated_
                        insideConstrainedOut = d_15_closedInside_
                        currentConstrainedOut = d_16_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_3_rollbackLimit_):
                        d_17_rolledGenerated_: _dafny.Seq
                        d_18_rolledCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out10_, out11_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_17_rolledGenerated_ = out10_
                        d_18_rolledCurrent_ = out11_
                        generated = d_17_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_18_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_19_stablePrefix_: _dafny.Seq
                        d_19_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_20_constrainedPrompt_: _dafny.Seq
                        d_20_constrainedPrompt_ = (prompt) + (d_19_stablePrefix_)
                        d_21_validCount_: int
                        out12_: int
                        out12_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_21_validCount_ = out12_
                        if (d_21_validCount_) > (d_4_narrowThreshold_):
                            d_22_remainingIn_: int
                            d_22_remainingIn_ = (maxSteps) - (d_1_steps_)
                            d_23_symbolBudget_: int
                            if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_22_remainingIn_)):
                                d_23_symbolBudget_ = d_22_remainingIn_
                            elif True:
                                d_23_symbolBudget_ = stepTokenBudget
                            d_24_symbolGenerated_: _dafny.Seq
                            d_25_symbolOut_: _dafny.Seq
                            d_26_hitEos_: bool
                            d_27_stepsUsed_: int
                            out13_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: int
                            out13_, out14_, out15_, out16_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_20_constrainedPrompt_, generated, currentConstrainedOut, d_23_symbolBudget_, eosToken)
                            d_24_symbolGenerated_ = out13_
                            d_25_symbolOut_ = out14_
                            d_26_hitEos_ = out15_
                            d_27_stepsUsed_ = out16_
                            generated = d_24_symbolGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_25_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_27_stepsUsed_)
                            if d_26_hitEos_:
                                raise _dafny.Break("0")
                        elif True:
                            d_28_nextIn_: _dafny.Seq
                            d_28_nextIn_ = eosToken
                            if (len(currentConstrainedOut)) < (2):
                                out17_: _dafny.Seq
                                out17_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('6e0'), eosToken)
                                d_28_nextIn_ = out17_
                            elif True:
                                out18_: _dafny.Seq
                                out18_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_28_nextIn_ = out18_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_28_nextIn_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_29_appendedGenerated_: _dafny.Seq
                                d_30_appendedInside_: bool
                                d_31_appendedCurrent_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_28_nextIn_)
                                d_29_appendedGenerated_ = out19_
                                d_30_appendedInside_ = out20_
                                d_31_appendedCurrent_ = out21_
                                generated = d_29_appendedGenerated_
                                insideConstrainedOut = d_30_appendedInside_
                                currentConstrainedOut = d_31_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

