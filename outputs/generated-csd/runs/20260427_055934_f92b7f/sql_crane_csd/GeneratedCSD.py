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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, eosToken):
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
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        d_3_minValidCount_: int
        d_3_minValidCount_ = 2
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_chunkBudget_: int
                        d_4_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_4_chunkBudget_) > (0):
                            d_5_chunkedGenerated_: _dafny.Seq
                            d_6_stoppedOnOpen_: bool
                            d_7_stoppedOnEos_: bool
                            d_8_stepsUsed_: int
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: bool
                            out3_: int
                            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_5_chunkedGenerated_ = out0_
                            d_6_stoppedOnOpen_ = out1_
                            d_7_stoppedOnEos_ = out2_
                            d_8_stepsUsed_ = out3_
                            generated = d_5_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                            if d_7_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_6_stoppedOnOpen_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_9_closedGenerated_: _dafny.Seq
                            d_10_closedInside_: bool
                            d_11_closedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_9_closedGenerated_ = out4_
                            d_10_closedInside_ = out5_
                            d_11_closedCurrent_ = out6_
                            generated = d_9_closedGenerated_
                            insideConstrainedOut = d_10_closedInside_
                            currentConstrainedOut = d_11_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_12_stablePrefix_: _dafny.Seq
                            d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_13_constrainedPrompt_: _dafny.Seq
                            d_13_constrainedPrompt_ = (prompt) + (d_12_stablePrefix_)
                            d_14_narrow_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, d_3_minValidCount_)
                            d_14_narrow_ = out7_
                            d_15_validCount_: int
                            out8_: int
                            out8_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_15_validCount_ = out8_
                            if (((d_14_narrow_) or ((d_15_validCount_) <= (d_2_narrowThreshold_))) or ((stepTokenBudget) <= (1))) or ((stepTokenBudget) > ((maxSteps) - (d_1_steps_))):
                                d_16_next_: _dafny.Seq
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_16_next_ = out9_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_16_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_17_appendedGenerated_: _dafny.Seq
                                    d_18_appendedInside_: bool
                                    d_19_appendedCurrent_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: _dafny.Seq
                                    out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                                    d_17_appendedGenerated_ = out10_
                                    d_18_appendedInside_ = out11_
                                    d_19_appendedCurrent_ = out12_
                                    generated = d_17_appendedGenerated_
                                    insideConstrainedOut = d_18_appendedInside_
                                    currentConstrainedOut = d_19_appendedCurrent_
                            elif True:
                                d_20_symbolOut_: _dafny.Seq
                                d_21_hitEos_: bool
                                d_22_stepsUsed_: int
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: int
                                out13_, out14_, out15_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, stepTokenBudget, eosToken)
                                d_20_symbolOut_ = out13_
                                d_21_hitEos_ = out14_
                                d_22_stepsUsed_ = out15_
                                generated = (d_12_stablePrefix_) + (d_20_symbolOut_)
                                insideConstrainedOut = True
                                currentConstrainedOut = d_20_symbolOut_
                                d_1_steps_ = (d_1_steps_) + (d_22_stepsUsed_)
                                if d_21_hitEos_:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

