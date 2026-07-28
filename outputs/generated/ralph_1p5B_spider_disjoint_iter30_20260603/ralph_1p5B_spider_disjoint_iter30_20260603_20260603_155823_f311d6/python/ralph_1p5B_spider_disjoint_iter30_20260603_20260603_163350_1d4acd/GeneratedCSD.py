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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate SQL: <<query>> using only the exact table and column names from the schema. Write the simplest correct query. Do not use extra joins or filters not required by the question.")))
        if ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_2_chunkMax_: int
            d_2_chunkMax_ = 8
            if (d_2_chunkMax_) > ((maxSteps) - (d_1_steps_)):
                d_2_chunkMax_ = (maxSteps) - (d_1_steps_)
            if (d_2_chunkMax_) > (0):
                d_3_chunkGenerated_: _dafny.Seq
                d_4_stoppedOnOpen_: bool
                d_5_stoppedOnEos_: bool
                d_6_chunkSteps_: int
                out0_: _dafny.Seq
                out1_: bool
                out2_: bool
                out3_: int
                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkMax_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                d_3_chunkGenerated_ = out0_
                d_4_stoppedOnOpen_ = out1_
                d_5_stoppedOnEos_ = out2_
                d_6_chunkSteps_ = out3_
                generated = d_3_chunkGenerated_
                d_1_steps_ = (d_1_steps_) + (d_6_chunkSteps_)
                if d_5_stoppedOnEos_:
                    cost = d_1_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                if d_4_stoppedOnOpen_:
                    d_7_openGenerated_: _dafny.Seq
                    d_8_openInside_: bool
                    d_9_openCurrent_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                    d_7_openGenerated_ = out4_
                    d_8_openInside_ = out5_
                    d_9_openCurrent_ = out6_
                    generated = d_7_openGenerated_
                    insideConstrainedOut = d_8_openInside_
                    currentConstrainedOut = d_9_openCurrent_
        if ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
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
            d_1_steps_ = (d_1_steps_) + (1)
        d_13_consecutiveDeadEnds_: int
        d_13_consecutiveDeadEnds_ = 0
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if (parser).IsCompletePrefix(currentConstrainedOut):
                        d_14_closedGenerated_: _dafny.Seq
                        d_15_closedInside_: bool
                        d_16_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_14_closedGenerated_ = out10_
                        d_15_closedInside_ = out11_
                        d_16_closedCurrent_ = out12_
                        generated = d_14_closedGenerated_
                        insideConstrainedOut = d_15_closedInside_
                        currentConstrainedOut = d_16_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_13_consecutiveDeadEnds_ = 0
                    elif True:
                        d_17_isDeadEnd_: bool
                        out13_: bool
                        out13_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_17_isDeadEnd_ = out13_
                        if (d_17_isDeadEnd_) and ((len(currentConstrainedOut)) > (0)):
                            d_18_rolledGenerated_: _dafny.Seq
                            d_19_rolledCurrent_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: _dafny.Seq
                            out14_, out15_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_18_rolledGenerated_ = out14_
                            d_19_rolledCurrent_ = out15_
                            generated = d_18_rolledGenerated_
                            currentConstrainedOut = d_19_rolledCurrent_
                            d_13_consecutiveDeadEnds_ = (d_13_consecutiveDeadEnds_) + (1)
                            if (d_13_consecutiveDeadEnds_) >= (3):
                                raise _dafny.Break("0")
                        elif True:
                            d_20_constrainedPrompt_: _dafny.Seq
                            d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_21_next_: _dafny.Seq
                            d_22_wasConstrained_: bool
                            out16_: _dafny.Seq
                            out17_: bool
                            out16_, out17_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_21_next_ = out16_
                            d_22_wasConstrained_ = out17_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_21_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_23_appendedGenerated_: _dafny.Seq
                                d_24_appendedInside_: bool
                                d_25_appendedCurrent_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                d_23_appendedGenerated_ = out18_
                                d_24_appendedInside_ = out19_
                                d_25_appendedCurrent_ = out20_
                                generated = d_23_appendedGenerated_
                                insideConstrainedOut = d_24_appendedInside_
                                currentConstrainedOut = d_25_appendedCurrent_
                                d_13_consecutiveDeadEnds_ = 0
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

