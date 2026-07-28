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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly: SQL: <<YOUR QUERY>> where YOUR QUERY is a single valid SQL query using the provided schema. No explanation, no markdown, no extra text."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        if ((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_3_chunkGenerated_: _dafny.Seq
            d_4_stoppedOnOpenSpan_: bool
            d_5_stoppedOnEos_: bool
            d_6_chunkSteps_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, (maxSteps) - (d_2_steps_), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_3_chunkGenerated_ = out0_
            d_4_stoppedOnOpenSpan_ = out1_
            d_5_stoppedOnEos_ = out2_
            d_6_chunkSteps_ = out3_
            d_2_steps_ = (d_2_steps_) + (d_6_chunkSteps_)
            generated = d_3_chunkGenerated_
            if d_5_stoppedOnEos_:
                cost = d_2_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            if d_4_stoppedOnOpenSpan_:
                d_7_enteredGenerated_: _dafny.Seq
                d_8_enteredInside_: bool
                d_9_enteredCurrent_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_7_enteredGenerated_ = out4_
                d_8_enteredInside_ = out5_
                d_9_enteredCurrent_ = out6_
                generated = d_7_enteredGenerated_
                insideConstrainedOut = d_8_enteredInside_
                currentConstrainedOut = d_9_enteredCurrent_
            elif True:
                if (d_2_steps_) < (maxSteps):
                    d_10_openGenerated_: _dafny.Seq
                    d_11_openInside_: bool
                    d_12_openCurrent_: _dafny.Seq
                    out7_: _dafny.Seq
                    out8_: bool
                    out9_: _dafny.Seq
                    out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_10_openGenerated_ = out7_
                    d_11_openInside_ = out8_
                    d_12_openCurrent_ = out9_
                    generated = d_10_openGenerated_
                    insideConstrainedOut = d_11_openInside_
                    currentConstrainedOut = d_12_openCurrent_
                    d_2_steps_ = (d_2_steps_) + (1)
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_13_closedGenerated_: _dafny.Seq
                        d_14_closedInside_: bool
                        d_15_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_13_closedGenerated_ = out10_
                        d_14_closedInside_ = out11_
                        d_15_closedCurrent_ = out12_
                        generated = d_13_closedGenerated_
                        insideConstrainedOut = d_14_closedInside_
                        currentConstrainedOut = d_15_closedCurrent_
                        d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_16_constrainedPrompt_: _dafny.Seq
                        d_16_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_17_next_: _dafny.Seq
                        d_18_wasConstrained_: bool
                        out13_: _dafny.Seq
                        out14_: bool
                        out13_, out14_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_17_next_ = out13_
                        d_18_wasConstrained_ = out14_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_17_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_19_appendedGenerated_: _dafny.Seq
                            d_20_appendedInside_: bool
                            d_21_appendedCurrent_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                            d_19_appendedGenerated_ = out15_
                            d_20_appendedInside_ = out16_
                            d_21_appendedCurrent_ = out17_
                            generated = d_19_appendedGenerated_
                            insideConstrainedOut = d_20_appendedInside_
                            currentConstrainedOut = d_21_appendedCurrent_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

