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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. For each arithmetic step, write a<op>b=c wrapped in << >>, e.g. <<3+4=7>>. Always close every << with >>. End the answer with: #### <integer>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanTokensInside_: int
        d_2_spanTokensInside_ = 0
        d_3_insideCap_: int
        d_3_insideCap_ = 18
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_chunkBudget_: int
                        d_4_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_4_chunkBudget_) > (32):
                            d_4_chunkBudget_ = 32
                        d_5_chunkedG_: _dafny.Seq
                        d_6_stoppedOpen_: bool
                        d_7_stoppedEos_: bool
                        d_8_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_5_chunkedG_ = out0_
                        d_6_stoppedOpen_ = out1_
                        d_7_stoppedEos_ = out2_
                        d_8_stepsUsed_ = out3_
                        generated = d_5_chunkedG_
                        d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                        if d_7_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_6_stoppedOpen_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_2_spanTokensInside_ = 0
                        elif (d_8_stepsUsed_) == (0):
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
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
                        d_2_spanTokensInside_ = 0
                    elif (d_2_spanTokensInside_) >= (d_3_insideCap_):
                        d_12_rolledGenerated_: _dafny.Seq
                        d_13_rolledCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: _dafny.Seq
                        out7_, out8_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_12_rolledGenerated_ = out7_
                        d_13_rolledCurrent_ = out8_
                        generated = d_12_rolledGenerated_
                        currentConstrainedOut = d_13_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanTokensInside_ = 0
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_14_closedGenerated_: _dafny.Seq
                            d_15_closedInside_: bool
                            d_16_closedCurrent_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_14_closedGenerated_ = out9_
                            d_15_closedInside_ = out10_
                            d_16_closedCurrent_ = out11_
                            generated = d_14_closedGenerated_
                            insideConstrainedOut = d_15_closedInside_
                            currentConstrainedOut = d_16_closedCurrent_
                            if (d_1_steps_) < (maxSteps):
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                    elif True:
                        d_17_stablePrefix_: _dafny.Seq
                        d_17_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (d_17_stablePrefix_)
                        d_19_remaining_: int
                        d_19_remaining_ = (maxSteps) - (d_1_steps_)
                        d_20_symbolBudget_: int
                        if (d_19_remaining_) > (8):
                            d_20_symbolBudget_ = 8
                        elif True:
                            d_20_symbolBudget_ = d_19_remaining_
                        if (d_20_symbolBudget_) == (0):
                            raise _dafny.Break("0")
                        d_21_symbolGenerated_: _dafny.Seq
                        d_22_symbolOut_: _dafny.Seq
                        d_23_hitEos_: bool
                        d_24_stepsUsed_: int
                        out12_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: bool
                        out15_: int
                        out12_, out13_, out14_, out15_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_18_constrainedPrompt_, generated, currentConstrainedOut, d_20_symbolBudget_, eosToken)
                        d_21_symbolGenerated_ = out12_
                        d_22_symbolOut_ = out13_
                        d_23_hitEos_ = out14_
                        d_24_stepsUsed_ = out15_
                        generated = d_21_symbolGenerated_
                        currentConstrainedOut = d_22_symbolOut_
                        d_1_steps_ = (d_1_steps_) + (d_24_stepsUsed_)
                        d_2_spanTokensInside_ = (d_2_spanTokensInside_) + (d_24_stepsUsed_)
                        if d_23_hitEos_:
                            raise _dafny.Break("0")
                        if (d_24_stepsUsed_) == (0):
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

