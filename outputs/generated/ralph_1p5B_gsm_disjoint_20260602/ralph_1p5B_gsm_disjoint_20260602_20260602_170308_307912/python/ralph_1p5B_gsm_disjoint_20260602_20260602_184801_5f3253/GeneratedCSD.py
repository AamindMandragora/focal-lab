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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Write your reasoning, then put the final numeric answer inside << >> delimiters.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeTokenCount_: int
        d_2_freeTokenCount_ = 0
        d_3_spanTokens_: int
        d_3_spanTokens_ = 0
        d_4_maxSpanTokens_: int
        d_4_maxSpanTokens_ = 20
        d_5_forceSpanThreshold_: int
        d_5_forceSpanThreshold_ = 60
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_2_freeTokenCount_) >= (d_5_forceSpanThreshold_)) and (((d_1_steps_) + (3)) <= (maxSteps)):
                            d_6_openGenerated_: _dafny.Seq
                            d_7_openInside_: bool
                            d_8_openCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_openGenerated_ = out0_
                            d_7_openInside_ = out1_
                            d_8_openCurrent_ = out2_
                            generated = d_6_openGenerated_
                            insideConstrainedOut = d_7_openInside_
                            currentConstrainedOut = d_8_openCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_3_spanTokens_ = 0
                            d_2_freeTokenCount_ = 0
                        elif True:
                            d_9_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_9_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                if ((d_2_freeTokenCount_) > (0)) and (((d_1_steps_) + (3)) <= (maxSteps)):
                                    d_10_openGenerated_: _dafny.Seq
                                    d_11_openInside_: bool
                                    d_12_openCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_10_openGenerated_ = out4_
                                    d_11_openInside_ = out5_
                                    d_12_openCurrent_ = out6_
                                    generated = d_10_openGenerated_
                                    insideConstrainedOut = d_11_openInside_
                                    currentConstrainedOut = d_12_openCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_3_spanTokens_ = 0
                                    d_2_freeTokenCount_ = 0
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                if (d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_3_spanTokens_ = 0
                                    d_2_freeTokenCount_ = 0
                                elif True:
                                    d_2_freeTokenCount_ = (d_2_freeTokenCount_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_13_closedGenerated_: _dafny.Seq
                        d_14_closedInside_: bool
                        d_15_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_13_closedGenerated_ = out7_
                        d_14_closedInside_ = out8_
                        d_15_closedCurrent_ = out9_
                        generated = d_13_closedGenerated_
                        insideConstrainedOut = d_14_closedInside_
                        currentConstrainedOut = d_15_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_3_spanTokens_ = 0
                        d_2_freeTokenCount_ = 0
                    elif True:
                        d_16_constrainedPrompt_: _dafny.Seq
                        d_16_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_17_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_17_next_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_17_next_) == (eosToken):
                            d_18_rolledGenerated_: _dafny.Seq
                            d_19_rolledCurrent_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: _dafny.Seq
                            out11_, out12_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_18_rolledGenerated_ = out11_
                            d_19_rolledCurrent_ = out12_
                            generated = d_18_rolledGenerated_
                            currentConstrainedOut = d_19_rolledCurrent_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                d_20_closedGenerated_: _dafny.Seq
                                d_21_closedInside_: bool
                                d_22_closedCurrent_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_20_closedGenerated_ = out13_
                                d_21_closedInside_ = out14_
                                d_22_closedCurrent_ = out15_
                                generated = d_20_closedGenerated_
                                insideConstrainedOut = d_21_closedInside_
                                currentConstrainedOut = d_22_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_3_spanTokens_ = 0
                                d_2_freeTokenCount_ = 0
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                        elif True:
                            d_23_appendedGenerated_: _dafny.Seq
                            d_24_appendedInside_: bool
                            d_25_appendedCurrent_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                            d_23_appendedGenerated_ = out16_
                            d_24_appendedInside_ = out17_
                            d_25_appendedCurrent_ = out18_
                            generated = d_23_appendedGenerated_
                            insideConstrainedOut = d_24_appendedInside_
                            currentConstrainedOut = d_25_appendedCurrent_
                            d_3_spanTokens_ = (d_3_spanTokens_) + (1)
                            if (d_3_spanTokens_) >= (d_4_maxSpanTokens_):
                                d_26_rolledGenerated_: _dafny.Seq
                                d_27_rolledCurrent_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: _dafny.Seq
                                out19_, out20_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_26_rolledGenerated_ = out19_
                                d_27_rolledCurrent_ = out20_
                                generated = d_26_rolledGenerated_
                                currentConstrainedOut = d_27_rolledCurrent_
                                d_3_spanTokens_ = 0
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                    d_28_closedGenerated_: _dafny.Seq
                                    d_29_closedInside_: bool
                                    d_30_closedCurrent_: _dafny.Seq
                                    out21_: _dafny.Seq
                                    out22_: bool
                                    out23_: _dafny.Seq
                                    out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_28_closedGenerated_ = out21_
                                    d_29_closedInside_ = out22_
                                    d_30_closedCurrent_ = out23_
                                    generated = d_28_closedGenerated_
                                    insideConstrainedOut = d_29_closedInside_
                                    currentConstrainedOut = d_30_closedCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    d_2_freeTokenCount_ = 0
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

