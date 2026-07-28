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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Show each intermediate symbolic calculation inside << >> delimiters. Put the final numeric answer inside << >> delimiters at the end. Example: The total is <<3 + 4 = 7>>, so the answer is <<7>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 20
        d_3_noisePenaltyTokens_: _dafny.Seq
        d_3_noisePenaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!!!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "...")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "~")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "@")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "#")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "$")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "%")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "^")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "&")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_5_newGenerated_: _dafny.Seq
                                d_6_newInside_: bool
                                d_7_newCurrent_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_5_newGenerated_ = out1_
                                d_6_newInside_ = out2_
                                d_7_newCurrent_ = out3_
                                generated = d_5_newGenerated_
                                insideConstrainedOut = d_6_newInside_
                                currentConstrainedOut = d_7_newCurrent_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_8_closedGenerated_: _dafny.Seq
                        d_9_closedInside_: bool
                        d_10_closedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_8_closedGenerated_ = out4_
                        d_9_closedInside_ = out5_
                        d_10_closedCurrent_ = out6_
                        generated = d_8_closedGenerated_
                        insideConstrainedOut = d_9_closedInside_
                        currentConstrainedOut = d_10_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_11_isDeadEnd_: bool
                        out7_: bool
                        out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_11_isDeadEnd_ = out7_
                        if d_11_isDeadEnd_:
                            d_12_rolledGenerated_: _dafny.Seq
                            d_13_rolledCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: _dafny.Seq
                            out8_, out9_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_12_rolledGenerated_ = out8_
                            d_13_rolledCurrent_ = out9_
                            generated = d_12_rolledGenerated_
                            currentConstrainedOut = d_13_rolledCurrent_
                            d_14_stillDeadEnd_: bool
                            out10_: bool
                            out10_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_14_stillDeadEnd_ = out10_
                            if d_14_stillDeadEnd_:
                                d_15_constrainedPrompt2_: _dafny.Seq
                                d_15_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_16_nextFallback_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_15_constrainedPrompt2_, currentConstrainedOut, generated, _dafny.BigRational('3e0'), eosToken)
                                d_16_nextFallback_ = out11_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_16_nextFallback_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_17_ag_: _dafny.Seq
                                    d_18_ai_: bool
                                    d_19_ac_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_nextFallback_)
                                    d_17_ag_ = out12_
                                    d_18_ai_ = out13_
                                    d_19_ac_ = out14_
                                    generated = d_17_ag_
                                    insideConstrainedOut = d_18_ai_
                                    currentConstrainedOut = d_19_ac_
                            elif True:
                                d_20_constrainedPrompt3_: _dafny.Seq
                                d_20_constrainedPrompt3_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_21_nextRep_: _dafny.Seq
                                out15_: _dafny.Seq
                                out15_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_20_constrainedPrompt3_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                d_21_nextRep_ = out15_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_21_nextRep_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_22_ag2_: _dafny.Seq
                                    d_23_ai2_: bool
                                    d_24_ac2_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_nextRep_)
                                    d_22_ag2_ = out16_
                                    d_23_ai2_ = out17_
                                    d_24_ac2_ = out18_
                                    generated = d_22_ag2_
                                    insideConstrainedOut = d_23_ai2_
                                    currentConstrainedOut = d_24_ac2_
                        elif True:
                            d_25_constrainedPrompt_: _dafny.Seq
                            d_25_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_26_next_: _dafny.Seq
                            out19_: _dafny.Seq
                            out19_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_25_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_3_noisePenaltyTokens_, _dafny.BigRational('8e0'), d_2_narrowThreshold_, eosToken)
                            d_26_next_ = out19_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_26_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_27_appendedGenerated_: _dafny.Seq
                                d_28_appendedInside_: bool
                                d_29_appendedCurrent_: _dafny.Seq
                                out20_: _dafny.Seq
                                out21_: bool
                                out22_: _dafny.Seq
                                out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next_)
                                d_27_appendedGenerated_ = out20_
                                d_28_appendedInside_ = out21_
                                d_29_appendedCurrent_ = out22_
                                generated = d_27_appendedGenerated_
                                insideConstrainedOut = d_28_appendedInside_
                                currentConstrainedOut = d_29_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

