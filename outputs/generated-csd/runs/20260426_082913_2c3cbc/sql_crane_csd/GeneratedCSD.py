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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, eosToken):
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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_chunkBudget_: int
                        d_2_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_3_chunkedGenerated_: _dafny.Seq
                        d_4_stoppedOnOpen_: bool
                        d_5_stoppedOnEos_: bool
                        d_6_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_3_chunkedGenerated_ = out0_
                        d_4_stoppedOnOpen_ = out1_
                        d_5_stoppedOnEos_ = out2_
                        d_6_stepsUsed_ = out3_
                        generated = d_3_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed_)
                        if d_5_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_4_stoppedOnOpen_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif (d_1_steps_) < (maxSteps):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_7_complete_: bool
                        d_7_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_7_complete_:
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
                            d_11_constrainedPrompt_: _dafny.Seq
                            d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_12_narrow_: bool
                            out7_: bool
                            out7_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 3)
                            d_12_narrow_ = out7_
                            if d_12_narrow_:
                                d_13_next_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_13_next_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_13_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_14_appendedGenerated1_: _dafny.Seq
                                    d_15_appendedInside1_: bool
                                    d_16_appendedCurrent1_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                                    d_14_appendedGenerated1_ = out9_
                                    d_15_appendedInside1_ = out10_
                                    d_16_appendedCurrent1_ = out11_
                                    generated = d_14_appendedGenerated1_
                                    insideConstrainedOut = d_15_appendedInside1_
                                    currentConstrainedOut = d_16_appendedCurrent1_
                            elif True:
                                d_17_candidates_: _dafny.Seq
                                out12_: _dafny.Seq
                                out12_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, 4, eosToken)
                                d_17_candidates_ = out12_
                                (lm).GenerateLogits((d_11_constrainedPrompt_) + (currentConstrainedOut))
                                (d_0_helpers_).BoostTokenLogits(lm, d_17_candidates_, _dafny.BigRational('8e0'))
                                d_18_next_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = (lm).ChooseNextTokenUnconstrained()
                                d_18_next_ = out13_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_18_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_19_valid_: bool
                                    out14_: bool
                                    out14_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_18_next_)
                                    d_19_valid_ = out14_
                                    if d_19_valid_:
                                        d_20_appendedGenerated2_: _dafny.Seq
                                        d_21_appendedInside2_: bool
                                        d_22_appendedCurrent2_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out16_: bool
                                        out17_: _dafny.Seq
                                        out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                                        d_20_appendedGenerated2_ = out15_
                                        d_21_appendedInside2_ = out16_
                                        d_22_appendedCurrent2_ = out17_
                                        generated = d_20_appendedGenerated2_
                                        insideConstrainedOut = d_21_appendedInside2_
                                        currentConstrainedOut = d_22_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

