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
        d_2_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_2_flatGroups_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_chunkBudget_: int
                        d_3_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_4_chunkedGenerated_: _dafny.Seq
                        d_5_stoppedOnOpen_: bool
                        d_6_stoppedOnEos_: bool
                        d_7_stepsUsed_: int
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: bool
                        out4_: int
                        out1_, out2_, out3_, out4_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_4_chunkedGenerated_ = out1_
                        d_5_stoppedOnOpen_ = out2_
                        d_6_stoppedOnEos_ = out3_
                        d_7_stepsUsed_ = out4_
                        generated = d_4_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                        if d_6_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_5_stoppedOnOpen_:
                            d_8_observedGenerated_: _dafny.Seq
                            d_9_observedInside_: bool
                            d_10_observedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_8_observedGenerated_ = out5_
                            d_9_observedInside_ = out6_
                            d_10_observedCurrent_ = out7_
                            generated = d_8_observedGenerated_
                            insideConstrainedOut = d_9_observedInside_
                            currentConstrainedOut = d_10_observedCurrent_
                        elif True:
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_11_closedGenerated_: _dafny.Seq
                        d_12_closedInside_: bool
                        d_13_closedCurrent_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_11_closedGenerated_ = out8_
                        d_12_closedInside_ = out9_
                        d_13_closedCurrent_ = out10_
                        generated = d_11_closedGenerated_
                        insideConstrainedOut = d_12_closedInside_
                        currentConstrainedOut = d_13_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_14_stablePrefix_: _dafny.Seq
                        d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (d_14_stablePrefix_)
                        d_16_validCount_: int
                        out11_: int
                        out11_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_16_validCount_ = out11_
                        if (d_16_validCount_) <= (12):
                            d_17_next_: _dafny.Seq
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_17_next_ = out12_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_17_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_18_appendedGenerated_: _dafny.Seq
                                d_19_appendedInside_: bool
                                d_20_appendedCurrent_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                d_18_appendedGenerated_ = out13_
                                d_19_appendedInside_ = out14_
                                d_20_appendedCurrent_ = out15_
                                generated = d_18_appendedGenerated_
                                insideConstrainedOut = d_19_appendedInside_
                                currentConstrainedOut = d_20_appendedCurrent_
                        elif True:
                            d_21_remaining_: int
                            d_21_remaining_ = (maxSteps) - (d_1_steps_)
                            d_22_symbolBudget_: int
                            if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_21_remaining_)):
                                d_22_symbolBudget_ = d_21_remaining_
                            elif True:
                                d_22_symbolBudget_ = stepTokenBudget
                            d_23_symbolGenerated_: _dafny.Seq
                            d_24_symbolCurrent_: _dafny.Seq
                            d_25_hitEos_: bool
                            d_26_symbolSteps_: int
                            out16_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: bool
                            out19_: int
                            out16_, out17_, out18_, out19_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_15_constrainedPrompt_, generated, currentConstrainedOut, d_22_symbolBudget_, eosToken)
                            d_23_symbolGenerated_ = out16_
                            d_24_symbolCurrent_ = out17_
                            d_25_hitEos_ = out18_
                            d_26_symbolSteps_ = out19_
                            generated = d_23_symbolGenerated_
                            currentConstrainedOut = d_24_symbolCurrent_
                            d_1_steps_ = (d_1_steps_) + (d_26_symbolSteps_)
                            if d_25_hitEos_:
                                raise _dafny.Break("0")
                            elif ((len(d_24_symbolCurrent_)) == (len(currentConstrainedOut))) and ((d_26_symbolSteps_) == (0)):
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

