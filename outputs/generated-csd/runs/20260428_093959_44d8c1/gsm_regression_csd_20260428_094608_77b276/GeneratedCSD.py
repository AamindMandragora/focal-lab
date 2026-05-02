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
        d_2_preferredFlat_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_2_preferredFlat_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_next_: _dafny.Seq
                        out1_: _dafny.Seq
                        out1_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_3_next_ = out1_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                            if VerifiedDecoderAgent.default__.Contains(d_3_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_4_openedGenerated_: _dafny.Seq
                                d_5_openedInside_: bool
                                d_6_openedCurrent_: _dafny.Seq
                                out2_: _dafny.Seq
                                out3_: bool
                                out4_: _dafny.Seq
                                out2_, out3_, out4_ = (d_0_helpers_).OpenConstrainedSpan(lm, _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (1):]))
                                d_4_openedGenerated_ = out2_
                                d_5_openedInside_ = out3_
                                d_6_openedCurrent_ = out4_
                                generated = d_4_openedGenerated_
                                insideConstrainedOut = d_5_openedInside_
                                currentConstrainedOut = d_6_openedCurrent_
                    elif True:
                        d_7_completeNow_: bool
                        d_7_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_7_completeNow_:
                            if ((d_1_steps_) + (1)) <= (maxSteps):
                                d_8_closedGenerated_: _dafny.Seq
                                d_9_closedInside_: bool
                                d_10_closedCurrent_: _dafny.Seq
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: _dafny.Seq
                                out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_8_closedGenerated_ = out5_
                                d_9_closedInside_ = out6_
                                d_10_closedCurrent_ = out7_
                                generated = d_8_closedGenerated_
                                insideConstrainedOut = d_9_closedInside_
                                currentConstrainedOut = d_10_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_11_stablePrefix_: _dafny.Seq
                                d_11_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_12_rolledGenerated_: _dafny.Seq
                                d_13_rolledCurrent_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: _dafny.Seq
                                out8_, out9_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_11_stablePrefix_, generated, currentConstrainedOut)
                                d_12_rolledGenerated_ = out8_
                                d_13_rolledCurrent_ = out9_
                                generated = d_12_rolledGenerated_
                                currentConstrainedOut = d_13_rolledCurrent_
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                        elif True:
                            if ((d_1_steps_) + (1)) >= (maxSteps):
                                d_14_stablePrefix2_: _dafny.Seq
                                d_14_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_15_rolledGenerated2_: _dafny.Seq
                                d_16_rolledCurrent2_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: _dafny.Seq
                                out10_, out11_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_14_stablePrefix2_, generated, currentConstrainedOut)
                                d_15_rolledGenerated2_ = out10_
                                d_16_rolledCurrent2_ = out11_
                                generated = d_15_rolledGenerated2_
                                currentConstrainedOut = d_16_rolledCurrent2_
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                            elif True:
                                d_17_dead_: bool
                                out12_: bool
                                out12_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                                d_17_dead_ = out12_
                                if d_17_dead_:
                                    d_18_stablePrefix3_: _dafny.Seq
                                    d_18_stablePrefix3_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    d_19_rolledGenerated3_: _dafny.Seq
                                    d_20_rolledCurrent3_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out13_, out14_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_18_stablePrefix3_, generated, currentConstrainedOut)
                                    d_19_rolledGenerated3_ = out13_
                                    d_20_rolledCurrent3_ = out14_
                                    generated = d_19_rolledGenerated3_
                                    currentConstrainedOut = d_20_rolledCurrent3_
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                elif True:
                                    d_21_stablePrefix4_: _dafny.Seq
                                    d_21_stablePrefix4_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    d_22_constrainedPrompt_: _dafny.Seq
                                    d_22_constrainedPrompt_ = (prompt) + (d_21_stablePrefix4_)
                                    (lm).GenerateLogits((d_22_constrainedPrompt_) + (currentConstrainedOut))
                                    if (len(d_2_preferredFlat_)) > (0):
                                        d_23_candidates_: _dafny.Seq
                                        out15_: _dafny.Seq
                                        out15_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, 16, eosToken)
                                        d_23_candidates_ = out15_
                                        d_24_preferred_: _dafny.Seq
                                        out16_: _dafny.Seq
                                        out16_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_23_candidates_, d_2_preferredFlat_)
                                        d_24_preferred_ = out16_
                                        if (len(d_24_preferred_)) > (0):
                                            (d_0_helpers_).BoostTokenLogits(lm, d_24_preferred_, _dafny.BigRational('8e0'))
                                    (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                    d_25_next2_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out17_ = (lm).ChooseNextToken()
                                    d_25_next2_ = out17_
                                    (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_25_next2_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_26_appendedGenerated_: _dafny.Seq
                                        d_27_appendedInside_: bool
                                        d_28_appendedCurrent_: _dafny.Seq
                                        out18_: _dafny.Seq
                                        out19_: bool
                                        out20_: _dafny.Seq
                                        out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next2_)
                                        d_26_appendedGenerated_ = out18_
                                        d_27_appendedInside_ = out19_
                                        d_28_appendedCurrent_ = out20_
                                        generated = d_26_appendedGenerated_
                                        insideConstrainedOut = d_27_appendedInside_
                                        currentConstrainedOut = d_28_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

