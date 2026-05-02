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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_boundaryToken_: _dafny.Seq
        d_2_boundaryToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))
        d_3_openSpanToken_: _dafny.Seq
        d_3_openSpanToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_chunkBudget_: int
                        d_4_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_5_generatedOut_: _dafny.Seq
                        d_6_stoppedOnOpenSpan_: bool
                        d_7_stoppedOnEos_: bool
                        d_8_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, d_3_openSpanToken_, eosToken)
                        d_5_generatedOut_ = out0_
                        d_6_stoppedOnOpenSpan_ = out1_
                        d_7_stoppedOnEos_ = out2_
                        d_8_stepsUsed_ = out3_
                        generated = d_5_generatedOut_
                        d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                        if d_7_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif True:
                            if d_6_stoppedOnOpenSpan_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_9_dead_: bool
                        out4_: bool
                        out4_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_9_dead_ = out4_
                        if d_9_dead_:
                            d_10_repaired_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_2_boundaryToken_)
                            d_10_repaired_ = out5_
                            generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - ((len(currentConstrainedOut)) - (len(d_10_repaired_))):])
                            currentConstrainedOut = d_10_repaired_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_11_isComplete_: bool
                            d_11_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_11_isComplete_:
                                if ((d_1_steps_) + (1)) < (maxSteps):
                                    d_12_stablePrefix0_: _dafny.Seq
                                    d_12_stablePrefix0_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    d_13_constrainedPrompt0_: _dafny.Seq
                                    d_13_constrainedPrompt0_ = (prompt) + (d_12_stablePrefix0_)
                                    (lm).GenerateLogits((d_13_constrainedPrompt0_) + (currentConstrainedOut))
                                    if (len(validTokenGroups)) > (0):
                                        (d_0_helpers_).BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, _dafny.BigRational('1e0'))
                                        d_14_flat0_: _dafny.Seq
                                        out6_: _dafny.Seq
                                        out6_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                        d_14_flat0_ = out6_
                                        (d_0_helpers_).PenalizeTokenLogits(lm, d_14_flat0_, _dafny.BigRational('2e-1'))
                                    (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('8e0'))
                                    (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                    d_15_next0_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out7_ = (lm).ChooseNextToken()
                                    d_15_next0_ = out7_
                                    (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_15_next0_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_16_generatedKeep_: _dafny.Seq
                                        d_17_insideKeep_: bool
                                        d_18_currentKeep_: _dafny.Seq
                                        out8_: _dafny.Seq
                                        out9_: bool
                                        out10_: _dafny.Seq
                                        out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next0_)
                                        d_16_generatedKeep_ = out8_
                                        d_17_insideKeep_ = out9_
                                        d_18_currentKeep_ = out10_
                                        generated = d_16_generatedKeep_
                                        insideConstrainedOut = d_17_insideKeep_
                                        currentConstrainedOut = d_18_currentKeep_
                                elif True:
                                    d_19_generated2_: _dafny.Seq
                                    d_20_inside2_: bool
                                    d_21_current2_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_19_generated2_ = out11_
                                    d_20_inside2_ = out12_
                                    d_21_current2_ = out13_
                                    generated = d_19_generated2_
                                    insideConstrainedOut = d_20_inside2_
                                    currentConstrainedOut = d_21_current2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_22_stablePrefix_: _dafny.Seq
                                d_22_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_23_constrainedPrompt_: _dafny.Seq
                                d_23_constrainedPrompt_ = (prompt) + (d_22_stablePrefix_)
                                (lm).GenerateLogits((d_23_constrainedPrompt_) + (currentConstrainedOut))
                                if (len(validTokenGroups)) > (0):
                                    (d_0_helpers_).BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, _dafny.BigRational('15e-1'))
                                    d_24_flat_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out14_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_24_flat_ = out14_
                                    (d_0_helpers_).PenalizeTokenLogits(lm, d_24_flat_, _dafny.BigRational('1e-1'))
                                (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('1e2'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_25_next_: _dafny.Seq
                                out15_: _dafny.Seq
                                out15_ = (lm).ChooseNextToken()
                                d_25_next_ = out15_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_25_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_26_generated3_: _dafny.Seq
                                    d_27_inside3_: bool
                                    d_28_current3_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next_)
                                    d_26_generated3_ = out16_
                                    d_27_inside3_ = out17_
                                    d_28_current3_ = out18_
                                    generated = d_26_generated3_
                                    insideConstrainedOut = d_27_inside3_
                                    currentConstrainedOut = d_28_current3_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

