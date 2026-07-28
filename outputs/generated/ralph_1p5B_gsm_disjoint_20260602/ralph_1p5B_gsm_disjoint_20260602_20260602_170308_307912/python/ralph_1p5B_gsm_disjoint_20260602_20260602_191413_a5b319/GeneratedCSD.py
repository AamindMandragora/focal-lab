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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. At the end, write only the final numeric answer (using numbers, +, -, *, /, and variable names without curly braces) inside << >> delimiters. Example: <<42>> or <<n * 3>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanTokens_: int
        d_2_spanTokens_ = 0
        d_3_maxSpanTokens_: int
        d_3_maxSpanTokens_ = 20
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
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_spanTokens_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_5_closedGenerated_: _dafny.Seq
                        d_6_closedInside_: bool
                        d_7_closedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_5_closedGenerated_ = out1_
                        d_6_closedInside_ = out2_
                        d_7_closedCurrent_ = out3_
                        generated = d_5_closedGenerated_
                        insideConstrainedOut = d_6_closedInside_
                        currentConstrainedOut = d_7_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanTokens_ = 0
                    elif True:
                        d_8_constrainedPrompt_: _dafny.Seq
                        d_8_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_9_next_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_9_next_ = out4_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_9_next_) == (eosToken):
                            d_10_rolledGenerated_: _dafny.Seq
                            d_11_rolledCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: _dafny.Seq
                            out5_, out6_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_10_rolledGenerated_ = out5_
                            d_11_rolledCurrent_ = out6_
                            generated = d_10_rolledGenerated_
                            currentConstrainedOut = d_11_rolledCurrent_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_12_closedGenerated_: _dafny.Seq
                                d_13_closedInside_: bool
                                d_14_closedCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_12_closedGenerated_ = out7_
                                d_13_closedInside_ = out8_
                                d_14_closedCurrent_ = out9_
                                generated = d_12_closedGenerated_
                                insideConstrainedOut = d_13_closedInside_
                                currentConstrainedOut = d_14_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_2_spanTokens_ = 0
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_spanTokens_ = 0
                        elif True:
                            d_15_valid_: bool
                            out10_: bool
                            out10_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_9_next_)
                            d_15_valid_ = out10_
                            if d_15_valid_:
                                d_16_appendedGenerated_: _dafny.Seq
                                d_17_appendedInside_: bool
                                d_18_appendedCurrent_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                                d_16_appendedGenerated_ = out11_
                                d_17_appendedInside_ = out12_
                                d_18_appendedCurrent_ = out13_
                                generated = d_16_appendedGenerated_
                                insideConstrainedOut = d_17_appendedInside_
                                currentConstrainedOut = d_18_appendedCurrent_
                                d_2_spanTokens_ = (d_2_spanTokens_) + (1)
                                if (d_2_spanTokens_) >= (d_3_maxSpanTokens_):
                                    d_19_rolledGenerated_: _dafny.Seq
                                    d_20_rolledCurrent_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out14_, out15_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                    d_19_rolledGenerated_ = out14_
                                    d_20_rolledCurrent_ = out15_
                                    generated = d_19_rolledGenerated_
                                    currentConstrainedOut = d_20_rolledCurrent_
                                    d_2_spanTokens_ = 0
                                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                        d_21_closedGenerated_: _dafny.Seq
                                        d_22_closedInside_: bool
                                        d_23_closedCurrent_: _dafny.Seq
                                        out16_: _dafny.Seq
                                        out17_: bool
                                        out18_: _dafny.Seq
                                        out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_21_closedGenerated_ = out16_
                                        d_22_closedInside_ = out17_
                                        d_23_closedCurrent_ = out18_
                                        generated = d_21_closedGenerated_
                                        insideConstrainedOut = d_22_closedInside_
                                        currentConstrainedOut = d_23_closedCurrent_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        insideConstrainedOut = False
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

