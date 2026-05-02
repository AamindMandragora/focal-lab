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
        d_2_sawOpen_: bool
        d_2_sawOpen_ = insideConstrained
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_rem0_: int
                        d_3_rem0_ = (maxSteps) - (d_1_steps_)
                        if (d_3_rem0_) == (0):
                            raise _dafny.Break("0")
                        elif True:
                            d_4_chunkGenerated_: _dafny.Seq
                            d_5_stoppedOnOpenSpan_: bool
                            d_6_stoppedOnEos_: bool
                            d_7_stepsUsed_: int
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: bool
                            out3_: int
                            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_rem0_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_4_chunkGenerated_ = out0_
                            d_5_stoppedOnOpenSpan_ = out1_
                            d_6_stoppedOnEos_ = out2_
                            d_7_stepsUsed_ = out3_
                            generated = d_4_chunkGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                            if d_6_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif True:
                                if d_5_stoppedOnOpenSpan_:
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_sawOpen_ = True
                                elif True:
                                    if d_2_sawOpen_:
                                        raise _dafny.Break("0")
                    elif True:
                        d_8_rem1_: int
                        d_8_rem1_ = (maxSteps) - (d_1_steps_)
                        if (d_8_rem1_) == (0):
                            raise _dafny.Break("0")
                        elif True:
                            d_9_complete_: bool
                            d_9_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_9_complete_:
                                d_10_closedGenerated_: _dafny.Seq
                                d_11_closedInside_: bool
                                d_12_closedCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_10_closedGenerated_ = out4_
                                d_11_closedInside_ = out5_
                                d_12_closedCurrent_ = out6_
                                generated = d_10_closedGenerated_
                                insideConstrainedOut = d_11_closedInside_
                                currentConstrainedOut = d_12_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_13_stablePrefix_: _dafny.Seq
                                d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_14_constrainedPrompt_: _dafny.Seq
                                d_14_constrainedPrompt_ = (prompt) + (d_13_stablePrefix_)
                                if (d_8_rem1_) == (1):
                                    d_15_symbolOut1_: _dafny.Seq
                                    d_16_hitEos1_: bool
                                    d_17_stepsUsed1_: int
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: int
                                    out7_, out8_, out9_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, d_8_rem1_, eosToken)
                                    d_15_symbolOut1_ = out7_
                                    d_16_hitEos1_ = out8_
                                    d_17_stepsUsed1_ = out9_
                                    generated = (d_13_stablePrefix_) + (d_15_symbolOut1_)
                                    insideConstrainedOut = True
                                    currentConstrainedOut = d_15_symbolOut1_
                                    d_1_steps_ = (d_1_steps_) + (d_17_stepsUsed1_)
                                    if d_16_hitEos1_:
                                        raise _dafny.Break("0")
                                elif True:
                                    d_18_symbolOut_: _dafny.Seq
                                    d_19_hitEos_: bool
                                    d_20_stepsUsed2_: int
                                    out10_: _dafny.Seq
                                    out11_: bool
                                    out12_: int
                                    out10_, out11_, out12_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, d_8_rem1_, eosToken)
                                    d_18_symbolOut_ = out10_
                                    d_19_hitEos_ = out11_
                                    d_20_stepsUsed2_ = out12_
                                    generated = (d_13_stablePrefix_) + (d_18_symbolOut_)
                                    insideConstrainedOut = True
                                    currentConstrainedOut = d_18_symbolOut_
                                    d_1_steps_ = (d_1_steps_) + (d_20_stepsUsed2_)
                                    if d_19_hitEos_:
                                        raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

