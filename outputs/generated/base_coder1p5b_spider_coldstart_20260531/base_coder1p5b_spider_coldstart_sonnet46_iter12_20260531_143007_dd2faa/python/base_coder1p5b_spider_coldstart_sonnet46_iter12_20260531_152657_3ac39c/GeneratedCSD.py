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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write a single concise SQL SELECT query. Output format: SQL: <<QUERY>>. Use only the schema tables and columns provided. No repeated conditions. Keep the query as short as possible.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_2_openedGenerated_: _dafny.Seq
            d_3_openedInside_: bool
            d_4_openedCurrent_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_2_openedGenerated_ = out0_
            d_3_openedInside_ = out1_
            d_4_openedCurrent_ = out2_
            generated = d_2_openedGenerated_
            insideConstrainedOut = d_3_openedInside_
            currentConstrainedOut = d_4_openedCurrent_
            d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if (parser).IsCompletePrefix(currentConstrainedOut):
                        d_5_closedGenerated_: _dafny.Seq
                        d_6_closedInside_: bool
                        d_7_closedCurrent_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_5_closedGenerated_ = out3_
                        d_6_closedInside_ = out4_
                        d_7_closedCurrent_ = out5_
                        generated = d_5_closedGenerated_
                        insideConstrainedOut = d_6_closedInside_
                        currentConstrainedOut = d_7_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        if (len(currentConstrainedOut)) > (80):
                            d_8_rolledGenerated_: _dafny.Seq
                            d_9_rolledCurrent_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: _dafny.Seq
                            out6_, out7_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_8_rolledGenerated_ = out6_
                            d_9_rolledCurrent_ = out7_
                            generated = d_8_rolledGenerated_
                            currentConstrainedOut = d_9_rolledCurrent_
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_10_closedGenerated_: _dafny.Seq
                                d_11_closedInside_: bool
                                d_12_closedCurrent_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_10_closedGenerated_ = out8_
                                d_11_closedInside_ = out9_
                                d_12_closedCurrent_ = out10_
                                generated = d_10_closedGenerated_
                                insideConstrainedOut = d_11_closedInside_
                                currentConstrainedOut = d_12_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_13_constrainedPrompt_: _dafny.Seq
                                d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_14_next_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('8e0'), eosToken)
                                d_14_next_ = out11_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_14_next_) == (eosToken):
                                    if (parser).IsCompletePrefix(currentConstrainedOut):
                                        pass
                                    elif True:
                                        raise _dafny.Break("0")
                                elif True:
                                    d_15_appendedGenerated_: _dafny.Seq
                                    d_16_appendedInside_: bool
                                    d_17_appendedCurrent_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                    d_15_appendedGenerated_ = out12_
                                    d_16_appendedInside_ = out13_
                                    d_17_appendedCurrent_ = out14_
                                    generated = d_15_appendedGenerated_
                                    insideConstrainedOut = d_16_appendedInside_
                                    currentConstrainedOut = d_17_appendedCurrent_
                        elif True:
                            d_18_constrainedPrompt_: _dafny.Seq
                            d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_19_next_: _dafny.Seq
                            out15_: _dafny.Seq
                            out15_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('6e0'), eosToken)
                            d_19_next_ = out15_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_19_next_) == (eosToken):
                                if (parser).IsCompletePrefix(currentConstrainedOut):
                                    pass
                                elif True:
                                    if (d_1_steps_) < (maxSteps):
                                        d_20_next2_: _dafny.Seq
                                        out16_: _dafny.Seq
                                        out16_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                        d_20_next2_ = out16_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        if (d_20_next2_) == (eosToken):
                                            raise _dafny.Break("0")
                                        elif True:
                                            d_21_appendedGenerated_: _dafny.Seq
                                            d_22_appendedInside_: bool
                                            d_23_appendedCurrent_: _dafny.Seq
                                            out17_: _dafny.Seq
                                            out18_: bool
                                            out19_: _dafny.Seq
                                            out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next2_)
                                            d_21_appendedGenerated_ = out17_
                                            d_22_appendedInside_ = out18_
                                            d_23_appendedCurrent_ = out19_
                                            generated = d_21_appendedGenerated_
                                            insideConstrainedOut = d_22_appendedInside_
                                            currentConstrainedOut = d_23_appendedCurrent_
                                    elif True:
                                        raise _dafny.Break("0")
                            elif True:
                                d_24_appendedGenerated_: _dafny.Seq
                                d_25_appendedInside_: bool
                                d_26_appendedCurrent_: _dafny.Seq
                                out20_: _dafny.Seq
                                out21_: bool
                                out22_: _dafny.Seq
                                out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                d_24_appendedGenerated_ = out20_
                                d_25_appendedInside_ = out21_
                                d_26_appendedCurrent_ = out22_
                                generated = d_24_appendedGenerated_
                                insideConstrainedOut = d_25_appendedInside_
                                currentConstrainedOut = d_26_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

