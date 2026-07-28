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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Produce a single SQL query in the format: SQL: <<query>>")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_warmup__steps__taken_: int
                        d_2_warmup__steps__taken_ = (len(generated)) - (len(generatedPrefix))
                        if (d_2_warmup__steps__taken_) < (5):
                            d_3_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_3_next_ = out0_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_3_next_) == (eosToken):
                                raise _dafny.Break("0")
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                            if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_4_enteredGenerated_: _dafny.Seq
                                d_5_enteredInside_: bool
                                d_6_enteredCurrent_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_4_enteredGenerated_ = out1_
                                d_5_enteredInside_ = out2_
                                d_6_enteredCurrent_ = out3_
                                insideConstrainedOut = d_5_enteredInside_
                                currentConstrainedOut = d_6_enteredCurrent_
                        elif True:
                            d_7_chunkBudget_: int
                            if (maxSteps) > (d_1_steps_):
                                d_7_chunkBudget_ = (maxSteps) - (d_1_steps_)
                            elif True:
                                d_7_chunkBudget_ = 0
                            d_8_chunkedG_: _dafny.Seq
                            d_9_stoppedOpen_: bool
                            d_10_stoppedEos_: bool
                            d_11_stepsUsed_: int
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: bool
                            out7_: int
                            out4_, out5_, out6_, out7_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_7_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_8_chunkedG_ = out4_
                            d_9_stoppedOpen_ = out5_
                            d_10_stoppedEos_ = out6_
                            d_11_stepsUsed_ = out7_
                            generated = d_8_chunkedG_
                            d_1_steps_ = (d_1_steps_) + (d_11_stepsUsed_)
                            if d_10_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_9_stoppedOpen_:
                                d_12_enteredGenerated_: _dafny.Seq
                                d_13_enteredInside_: bool
                                d_14_enteredCurrent_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_12_enteredGenerated_ = out8_
                                d_13_enteredInside_ = out9_
                                d_14_enteredCurrent_ = out10_
                                insideConstrainedOut = d_13_enteredInside_
                                currentConstrainedOut = d_14_enteredCurrent_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_15_closedGenerated_: _dafny.Seq
                        d_16_closedInside_: bool
                        d_17_closedCurrent_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: _dafny.Seq
                        out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_15_closedGenerated_ = out11_
                        d_16_closedInside_ = out12_
                        d_17_closedCurrent_ = out13_
                        generated = d_15_closedGenerated_
                        insideConstrainedOut = d_16_closedInside_
                        currentConstrainedOut = d_17_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_18_stablePrefix_: _dafny.Seq
                        d_18_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_19_constrainedPrompt_: _dafny.Seq
                        d_19_constrainedPrompt_ = (prompt) + (d_18_stablePrefix_)
                        d_20_symbolBudget_: int
                        d_20_symbolBudget_ = (maxSteps) - (d_1_steps_)
                        d_21_symbolGenerated_: _dafny.Seq
                        d_22_symbolOut_: _dafny.Seq
                        d_23_hitEos_: bool
                        d_24_stepsUsed_: int
                        out14_: _dafny.Seq
                        out15_: _dafny.Seq
                        out16_: bool
                        out17_: int
                        out14_, out15_, out16_, out17_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_19_constrainedPrompt_, generated, currentConstrainedOut, d_20_symbolBudget_, eosToken)
                        d_21_symbolGenerated_ = out14_
                        d_22_symbolOut_ = out15_
                        d_23_hitEos_ = out16_
                        d_24_stepsUsed_ = out17_
                        generated = d_21_symbolGenerated_
                        currentConstrainedOut = d_22_symbolOut_
                        d_1_steps_ = (d_1_steps_) + (d_24_stepsUsed_)
                        if d_23_hitEos_:
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

