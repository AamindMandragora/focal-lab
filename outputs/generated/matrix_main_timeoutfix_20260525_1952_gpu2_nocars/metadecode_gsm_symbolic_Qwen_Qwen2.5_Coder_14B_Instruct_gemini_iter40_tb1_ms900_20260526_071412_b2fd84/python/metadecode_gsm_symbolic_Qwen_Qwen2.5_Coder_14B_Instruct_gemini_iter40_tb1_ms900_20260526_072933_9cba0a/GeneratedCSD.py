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
        d_2_unconstrainedChunkBudget_: int
        d_2_unconstrainedChunkBudget_ = 32
        d_3_symbolChunkBudget_: int
        d_3_symbolChunkBudget_ = 10
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_remainingBudget_: int
                        d_4_remainingBudget_ = (maxSteps) - (d_1_steps_)
                        d_5_budgetForChunk_: int
                        if (d_2_unconstrainedChunkBudget_) > (d_4_remainingBudget_):
                            d_5_budgetForChunk_ = d_4_remainingBudget_
                        elif True:
                            d_5_budgetForChunk_ = d_2_unconstrainedChunkBudget_
                        if (d_5_budgetForChunk_) == (0):
                            raise _dafny.Break("0")
                        d_6_chunkedG_: _dafny.Seq
                        d_7_stoppedOpen_: bool
                        d_8_stoppedEos_: bool
                        d_9_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_budgetForChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_6_chunkedG_ = out0_
                        d_7_stoppedOpen_ = out1_
                        d_8_stoppedEos_ = out2_
                        d_9_stepsUsed_ = out3_
                        generated = d_6_chunkedG_
                        d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                        if d_8_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_7_stoppedOpen_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
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
                            d_15_remainingBudget_: int
                            d_15_remainingBudget_ = (maxSteps) - (d_1_steps_)
                            d_16_budgetForSymbol_: int
                            if (d_3_symbolChunkBudget_) > (d_15_remainingBudget_):
                                d_16_budgetForSymbol_ = d_15_remainingBudget_
                            elif True:
                                d_16_budgetForSymbol_ = d_3_symbolChunkBudget_
                            if (d_16_budgetForSymbol_) == (0):
                                raise _dafny.Break("0")
                            d_17_symbolGenerated_: _dafny.Seq
                            d_18_symbolOut_: _dafny.Seq
                            d_19_hitEos_: bool
                            d_20_stepsUsed_: int
                            out7_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: int
                            out7_, out8_, out9_, out10_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_14_constrainedPrompt_, generated, currentConstrainedOut, d_16_budgetForSymbol_, eosToken)
                            d_17_symbolGenerated_ = out7_
                            d_18_symbolOut_ = out8_
                            d_19_hitEos_ = out9_
                            d_20_stepsUsed_ = out10_
                            generated = d_17_symbolGenerated_
                            currentConstrainedOut = d_18_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_20_stepsUsed_)
                            if d_19_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

