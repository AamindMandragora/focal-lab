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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math problem step by step. For each arithmetic step write the computation inside << >>, like <<3+4=7>>. After all steps, on a new line write: #### <number> with no units, no commas, no extra text.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spansEmitted_: int
        d_2_spansEmitted_ = 0
        d_3_targetSpans_: int
        d_3_targetSpans_ = 4
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_4_closedGenerated_: _dafny.Seq
                            d_5_closedInside_: bool
                            d_6_closedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_4_closedGenerated_ = out0_
                            d_5_closedInside_ = out1_
                            d_6_closedCurrent_ = out2_
                            generated = d_4_closedGenerated_
                            insideConstrainedOut = d_5_closedInside_
                            currentConstrainedOut = d_6_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_spansEmitted_ = (d_2_spansEmitted_) + (1)
                        elif True:
                            d_7_stablePrefix_: _dafny.Seq
                            d_7_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_8_constrainedPrompt_: _dafny.Seq
                            d_8_constrainedPrompt_ = (prompt) + (d_7_stablePrefix_)
                            d_9_remaining_: int
                            d_9_remaining_ = (maxSteps) - (d_1_steps_)
                            d_10_symBudget_: int
                            if (d_9_remaining_) < (32):
                                d_10_symBudget_ = d_9_remaining_
                            elif True:
                                d_10_symBudget_ = 32
                            if (d_10_symBudget_) == (0):
                                raise _dafny.Break("0")
                            d_11_symbolGenerated_: _dafny.Seq
                            d_12_symbolOut_: _dafny.Seq
                            d_13_hitEos_: bool
                            d_14_stepsUsed_: int
                            out3_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_8_constrainedPrompt_, generated, currentConstrainedOut, d_10_symBudget_, eosToken)
                            d_11_symbolGenerated_ = out3_
                            d_12_symbolOut_ = out4_
                            d_13_hitEos_ = out5_
                            d_14_stepsUsed_ = out6_
                            generated = d_11_symbolGenerated_
                            currentConstrainedOut = d_12_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_14_stepsUsed_)
                            if d_13_hitEos_:
                                raise _dafny.Break("0")
                            if (d_14_stepsUsed_) == (0):
                                raise _dafny.Break("0")
                    elif True:
                        d_15_remaining_: int
                        d_15_remaining_ = (maxSteps) - (d_1_steps_)
                        if (d_15_remaining_) == (0):
                            raise _dafny.Break("0")
                        d_16_chunkBudget_: int
                        if (d_15_remaining_) < (60):
                            d_16_chunkBudget_ = d_15_remaining_
                        elif True:
                            d_16_chunkBudget_ = 60
                        d_17_chunkedG_: _dafny.Seq
                        d_18_stoppedOpen_: bool
                        d_19_stoppedEos_: bool
                        d_20_stepsUsed_: int
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: bool
                        out10_: int
                        out7_, out8_, out9_, out10_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_16_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_17_chunkedG_ = out7_
                        d_18_stoppedOpen_ = out8_
                        d_19_stoppedEos_ = out9_
                        d_20_stepsUsed_ = out10_
                        generated = d_17_chunkedG_
                        d_1_steps_ = (d_1_steps_) + (d_20_stepsUsed_)
                        if d_19_stoppedEos_:
                            raise _dafny.Break("0")
                        if d_18_stoppedOpen_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif (d_1_steps_) >= (maxSteps):
                            raise _dafny.Break("0")
                        elif (d_2_spansEmitted_) < (d_3_targetSpans_):
                            d_21_openedGenerated_: _dafny.Seq
                            d_22_openedInside_: bool
                            d_23_openedCurrent_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_21_openedGenerated_ = out11_
                            d_22_openedInside_ = out12_
                            d_23_openedCurrent_ = out13_
                            generated = d_21_openedGenerated_
                            insideConstrainedOut = d_22_openedInside_
                            currentConstrainedOut = d_23_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_24_next_: _dafny.Seq
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_24_next_ = out14_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_24_next_) == (eosToken):
                                raise _dafny.Break("0")
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_24_next_]))
                            if (d_24_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        if ((d_20_stepsUsed_) == (0)) and (not(d_18_stoppedOpen_)):
                            if (d_1_steps_) >= (maxSteps):
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

