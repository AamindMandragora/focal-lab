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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one SQL query in the required constrained span format. Do not add explanation, prose, or multiple alternatives.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 16
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_3_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                            if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_4_closedGenerated_: _dafny.Seq
                        d_5_closedInside_: bool
                        d_6_closedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_4_closedGenerated_ = out1_
                        d_5_closedInside_ = out2_
                        d_6_closedCurrent_ = out3_
                        generated = d_4_closedGenerated_
                        insideConstrainedOut = d_5_closedInside_
                        currentConstrainedOut = d_6_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_7_stablePrefix_: _dafny.Seq
                        d_7_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_8_constrainedPrompt_: _dafny.Seq
                        d_8_constrainedPrompt_ = (prompt) + (d_7_stablePrefix_)
                        d_9_broad_: bool
                        out4_: bool
                        out4_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, d_2_narrowThreshold_)
                        d_9_broad_ = out4_
                        if (not(d_9_broad_)) and ((stepTokenBudget) > (0)):
                            d_10_remaining_: int
                            d_10_remaining_ = (maxSteps) - (d_1_steps_)
                            d_11_symbolBudget_: int
                            if (stepTokenBudget) <= (d_10_remaining_):
                                d_11_symbolBudget_ = stepTokenBudget
                            elif True:
                                d_11_symbolBudget_ = d_10_remaining_
                            d_12_symbolGenerated_: _dafny.Seq
                            d_13_symbolOut_: _dafny.Seq
                            d_14_hitEos_: bool
                            d_15_stepsUsed_: int
                            out5_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: int
                            out5_, out6_, out7_, out8_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_8_constrainedPrompt_, generated, currentConstrainedOut, d_11_symbolBudget_, eosToken)
                            d_12_symbolGenerated_ = out5_
                            d_13_symbolOut_ = out6_
                            d_14_hitEos_ = out7_
                            d_15_stepsUsed_ = out8_
                            generated = d_12_symbolGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_13_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_15_stepsUsed_)
                            if d_14_hitEos_:
                                raise _dafny.Break("0")
                        elif True:
                            d_16_nextTok_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_16_nextTok_ = out9_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_16_nextTok_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_17_appendedGenerated_: _dafny.Seq
                                d_18_appendedInside_: bool
                                d_19_appendedCurrent_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_nextTok_)
                                d_17_appendedGenerated_ = out10_
                                d_18_appendedInside_ = out11_
                                d_19_appendedCurrent_ = out12_
                                generated = d_17_appendedGenerated_
                                insideConstrainedOut = d_18_appendedInside_
                                currentConstrainedOut = d_19_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

