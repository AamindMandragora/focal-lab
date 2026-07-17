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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer with exactly: SQL: <<YOUR QUERY>> where YOUR QUERY is a valid SQL SELECT statement. Use only tables and columns from the provided schema. Output nothing else.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_chunkBudget_: int
        d_2_chunkBudget_ = 8
        if (((d_1_steps_) + (d_2_chunkBudget_)) <= (maxSteps)) and (not(insideConstrainedOut)):
            d_3_chunkGenerated_: _dafny.Seq
            d_4_stoppedOnOpen_: bool
            d_5_stoppedOnEos_: bool
            d_6_chunkSteps_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_3_chunkGenerated_ = out0_
            d_4_stoppedOnOpen_ = out1_
            d_5_stoppedOnEos_ = out2_
            d_6_chunkSteps_ = out3_
            d_1_steps_ = (d_1_steps_) + (d_6_chunkSteps_)
            generated = d_3_chunkGenerated_
            if d_5_stoppedOnEos_:
                cost = d_1_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            if d_4_stoppedOnOpen_:
                insideConstrainedOut = True
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        if ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_7_openGenerated_: _dafny.Seq
            d_8_openInside_: bool
            d_9_openCurrent_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_7_openGenerated_ = out4_
            d_8_openInside_ = out5_
            d_9_openCurrent_ = out6_
            generated = d_7_openGenerated_
            insideConstrainedOut = d_8_openInside_
            currentConstrainedOut = d_9_openCurrent_
            d_1_steps_ = (d_1_steps_) + (1)
        d_10_longSpanCount_: int
        d_10_longSpanCount_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_11_closedGenerated_: _dafny.Seq
                        d_12_closedInside_: bool
                        d_13_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_11_closedGenerated_ = out7_
                        d_12_closedInside_ = out8_
                        d_13_closedCurrent_ = out9_
                        generated = d_11_closedGenerated_
                        insideConstrainedOut = d_12_closedInside_
                        currentConstrainedOut = d_13_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_14_constrainedPrompt_: _dafny.Seq
                        d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_15_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if (d_10_longSpanCount_) > (30):
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_15_next_ = out10_
                        elif True:
                            d_16_gatedNext_: _dafny.Seq
                            d_17_wasConstrained_: bool
                            out11_: _dafny.Seq
                            out12_: bool
                            out11_, out12_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_16_gatedNext_ = out11_
                            d_17_wasConstrained_ = out12_
                            d_15_next_ = d_16_gatedNext_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_15_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_18_appendedGenerated_: _dafny.Seq
                            d_19_appendedInside_: bool
                            d_20_appendedCurrent_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                            d_18_appendedGenerated_ = out13_
                            d_19_appendedInside_ = out14_
                            d_20_appendedCurrent_ = out15_
                            generated = d_18_appendedGenerated_
                            insideConstrainedOut = d_19_appendedInside_
                            currentConstrainedOut = d_20_appendedCurrent_
                            d_10_longSpanCount_ = (d_10_longSpanCount_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

