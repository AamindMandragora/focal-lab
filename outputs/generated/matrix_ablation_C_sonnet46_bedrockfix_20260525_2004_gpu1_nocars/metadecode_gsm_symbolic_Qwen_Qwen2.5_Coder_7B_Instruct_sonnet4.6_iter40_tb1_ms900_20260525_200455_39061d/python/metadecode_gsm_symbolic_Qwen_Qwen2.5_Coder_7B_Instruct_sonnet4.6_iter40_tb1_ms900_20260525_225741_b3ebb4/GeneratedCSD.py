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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step using plain text reasoning. You MUST include exactly ONE final expression inside << >> delimiters at the very end of your answer. Use variable names from the problem. Keep the expression concise (avoid repetition). Example: The answer is <<n1 * k + n2>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_maxConstrainedLen_: int
        d_2_maxConstrainedLen_ = 20
        d_3_spanSteps_: int
        d_3_spanSteps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_chunkBudget_: int
                        d_4_chunkBudget_ = (maxSteps) - (d_1_steps_)
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
                        if d_6_stoppedOpen_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_3_spanSteps_ = 0
                        elif d_7_stoppedEos_:
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
                        d_3_spanSteps_ = 0
                    elif ((d_3_spanSteps_) >= (d_2_maxConstrainedLen_)) or ((len(currentConstrainedOut)) >= (d_2_maxConstrainedLen_)):
                        insideConstrainedOut = False
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_spanSteps_ = 0
                    elif (((len(currentConstrainedOut)) >= (4)) and (((currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]) == ((currentConstrainedOut)[(len(currentConstrainedOut)) - (3)]))) and (((currentConstrainedOut)[(len(currentConstrainedOut)) - (2)]) == ((currentConstrainedOut)[(len(currentConstrainedOut)) - (4)])):
                        d_12_rolledGenerated_: _dafny.Seq
                        d_13_rolledCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: _dafny.Seq
                        out7_, out8_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_12_rolledGenerated_ = out7_
                        d_13_rolledCurrent_ = out8_
                        generated = d_12_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_13_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_spanSteps_ = (d_3_spanSteps_) + (1)
                    elif (((len(currentConstrainedOut)) >= (3)) and (((currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]) == ((currentConstrainedOut)[(len(currentConstrainedOut)) - (2)]))) and (((currentConstrainedOut)[(len(currentConstrainedOut)) - (2)]) == ((currentConstrainedOut)[(len(currentConstrainedOut)) - (3)])):
                        d_14_rolledGenerated_: _dafny.Seq
                        d_15_rolledCurrent_: _dafny.Seq
                        out9_: _dafny.Seq
                        out10_: _dafny.Seq
                        out9_, out10_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_14_rolledGenerated_ = out9_
                        d_15_rolledCurrent_ = out10_
                        generated = d_14_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_15_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_spanSteps_ = (d_3_spanSteps_) + (1)
                    elif True:
                        d_16_constrainedPrompt_: _dafny.Seq
                        d_16_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_17_next_: _dafny.Seq
                        out11_: _dafny.Seq
                        out11_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('4e0'), eosToken)
                        d_17_next_ = out11_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_spanSteps_ = (d_3_spanSteps_) + (1)
                        if (d_17_next_) == (eosToken):
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_3_spanSteps_ = 0
                        elif True:
                            d_18_appendedGenerated_: _dafny.Seq
                            d_19_appendedInside_: bool
                            d_20_appendedCurrent_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                            d_18_appendedGenerated_ = out12_
                            d_19_appendedInside_ = out13_
                            d_20_appendedCurrent_ = out14_
                            generated = d_18_appendedGenerated_
                            insideConstrainedOut = d_19_appendedInside_
                            currentConstrainedOut = d_20_appendedCurrent_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_21_closedGenerated_: _dafny.Seq
                d_22_closedInside_: bool
                d_23_closedCurrent_: _dafny.Seq
                out15_: _dafny.Seq
                out16_: bool
                out17_: _dafny.Seq
                out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_21_closedGenerated_ = out15_
                d_22_closedInside_ = out16_
                d_23_closedCurrent_ = out17_
                generated = d_21_closedGenerated_
                insideConstrainedOut = d_22_closedInside_
                currentConstrainedOut = d_23_closedCurrent_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

