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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Write each arithmetic expression inside << >> delimiters. After ####, write only the final numeric answer inside << >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanTokenCount_: int
        d_2_spanTokenCount_ = 0
        d_3_maxSpanTokens_: int
        d_3_maxSpanTokens_ = 15
        d_4_seenHash_: bool
        d_4_seenHash_ = False
        d_5_forcedFinalSpan_: bool
        d_5_forcedFinalSpan_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_4_seenHash_) and (not(d_5_forcedFinalSpan_))) and (((d_1_steps_) + (2)) < (maxSteps)):
                            d_6_g2_: _dafny.Seq
                            d_7_ins2_: bool
                            d_8_cur2_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_g2_ = out0_
                            d_7_ins2_ = out1_
                            d_8_cur2_ = out2_
                            generated = d_6_g2_
                            insideConstrainedOut = d_7_ins2_
                            currentConstrainedOut = d_8_cur2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_spanTokenCount_ = 0
                            d_5_forcedFinalSpan_ = True
                        elif True:
                            d_9_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_9_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif (d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                d_10_g2_: _dafny.Seq
                                d_11_ins2_: bool
                                d_12_cur2_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_10_g2_ = out4_
                                d_11_ins2_ = out5_
                                d_12_cur2_ = out6_
                                generated = d_10_g2_
                                insideConstrainedOut = d_11_ins2_
                                currentConstrainedOut = d_12_cur2_
                                d_2_spanTokenCount_ = 0
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                if (d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "####"))):
                                    d_4_seenHash_ = True
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
                        d_2_spanTokenCount_ = 0
                        if d_5_forcedFinalSpan_:
                            raise _dafny.Break("0")
                    elif (d_2_spanTokenCount_) >= (d_3_maxSpanTokens_):
                        d_16_rolledGenerated_: _dafny.Seq
                        d_17_rolledCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out10_, out11_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_16_rolledGenerated_ = out10_
                        d_17_rolledCurrent_ = out11_
                        generated = d_16_rolledGenerated_
                        currentConstrainedOut = d_17_rolledCurrent_
                        d_2_spanTokenCount_ = 0
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_18_closedGenerated_: _dafny.Seq
                            d_19_closedInside_: bool
                            d_20_closedCurrent_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_18_closedGenerated_ = out12_
                            d_19_closedInside_ = out13_
                            d_20_closedCurrent_ = out14_
                            generated = d_18_closedGenerated_
                            insideConstrainedOut = d_19_closedInside_
                            currentConstrainedOut = d_20_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if d_5_forcedFinalSpan_:
                                raise _dafny.Break("0")
                        elif True:
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_21_constrainedPrompt_: _dafny.Seq
                        d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_22_next_: _dafny.Seq
                        out15_: _dafny.Seq
                        out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_22_next_ = out15_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_22_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_23_appendedGenerated_: _dafny.Seq
                            d_24_appendedInside_: bool
                            d_25_appendedCurrent_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                            d_23_appendedGenerated_ = out16_
                            d_24_appendedInside_ = out17_
                            d_25_appendedCurrent_ = out18_
                            generated = d_23_appendedGenerated_
                            insideConstrainedOut = d_24_appendedInside_
                            currentConstrainedOut = d_25_appendedCurrent_
                            d_2_spanTokenCount_ = (d_2_spanTokenCount_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

