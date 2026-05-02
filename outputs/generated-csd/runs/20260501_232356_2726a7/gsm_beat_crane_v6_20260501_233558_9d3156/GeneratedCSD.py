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
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_2_flatGroups_ = out0_
        d_3_hasOpenedAny_: bool
        d_3_hasOpenedAny_ = (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in (generated)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        (lm).GenerateLogits((prompt) + (generated))
                        if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))) in ((lm).Tokens):
                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('1e2'))
                        if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens):
                            if d_3_hasOpenedAny_:
                                (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('2e1'))
                            elif True:
                                if (len(generated)) < (4):
                                    (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('4e0'))
                                elif True:
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('12e0'))
                        d_4_next_: _dafny.Seq
                        out1_: _dafny.Seq
                        out1_ = (lm).ChooseNextTokenUnconstrained()
                        d_4_next_ = out1_
                        (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_5_openedGenerated_: _dafny.Seq
                                d_6_openedInside_: bool
                                d_7_openedCurrent_: _dafny.Seq
                                out2_: _dafny.Seq
                                out3_: bool
                                out4_: _dafny.Seq
                                out2_, out3_, out4_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_5_openedGenerated_ = out2_
                                d_6_openedInside_ = out3_
                                d_7_openedCurrent_ = out4_
                                generated = d_5_openedGenerated_
                                insideConstrainedOut = d_6_openedInside_
                                currentConstrainedOut = d_7_openedCurrent_
                                d_3_hasOpenedAny_ = True
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                    elif True:
                        d_8_complete_: bool
                        d_8_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_8_complete_:
                            d_9_closedGenerated_: _dafny.Seq
                            d_10_closedInside_: bool
                            d_11_closedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_9_closedGenerated_ = out5_
                            d_10_closedInside_ = out6_
                            d_11_closedCurrent_ = out7_
                            generated = d_9_closedGenerated_
                            insideConstrainedOut = d_10_closedInside_
                            currentConstrainedOut = d_11_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            if ((len(currentConstrainedOut)) >= ((stepTokenBudget) + (3))) or ((len(currentConstrainedOut)) >= (12)):
                                d_12_repaired_: _dafny.Seq
                                d_13_excludedTok_: _dafny.Seq
                                d_14_hasExcluded_: bool
                                out8_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out8_, out9_, out10_ = VerifiedDecoderAgent.CSDHelpers.RollbackAndExclude(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")))
                                d_12_repaired_ = out8_
                                d_13_excludedTok_ = out9_
                                d_14_hasExcluded_ = out10_
                                generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - ((len(currentConstrainedOut)) - (len(d_12_repaired_))):])
                                currentConstrainedOut = d_12_repaired_
                                insideConstrainedOut = True
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_1_steps_) < (maxSteps):
                                    d_15_repairedComplete_: bool
                                    d_15_repairedComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if d_15_repairedComplete_:
                                        pass
                                    elif True:
                                        d_16_stablePrefix_: _dafny.Seq
                                        d_16_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                        if (d_14_hasExcluded_) and ((d_13_excludedTok_) in ((lm).Tokens)):
                                            d_17_penalizedNext_: _dafny.Seq
                                            out11_: _dafny.Seq
                                            out11_ = (d_0_helpers_).PenalizedConstrainedStep(lm, parser, (prompt) + (d_16_stablePrefix_), currentConstrainedOut, _dafny.SeqWithoutIsStrInference([d_13_excludedTok_]), _dafny.BigRational('8e0'), eosToken)
                                            d_17_penalizedNext_ = out11_
                                            if (d_17_penalizedNext_) == (eosToken):
                                                raise _dafny.Break("0")
                                            elif True:
                                                d_18_appendedGenerated1_: _dafny.Seq
                                                d_19_appendedInside1_: bool
                                                d_20_appendedCurrent1_: _dafny.Seq
                                                out12_: _dafny.Seq
                                                out13_: bool
                                                out14_: _dafny.Seq
                                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_penalizedNext_)
                                                d_18_appendedGenerated1_ = out12_
                                                d_19_appendedInside1_ = out13_
                                                d_20_appendedCurrent1_ = out14_
                                                generated = d_18_appendedGenerated1_
                                                insideConstrainedOut = d_19_appendedInside1_
                                                currentConstrainedOut = d_20_appendedCurrent1_
                            elif True:
                                d_21_stablePrefix2_: _dafny.Seq
                                d_21_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_22_next2_: _dafny.Seq
                                out15_: _dafny.Seq
                                out15_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_21_stablePrefix2_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('8e0'), 12, eosToken)
                                d_22_next2_ = out15_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_22_next2_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_23_appendedGenerated2_: _dafny.Seq
                                    d_24_appendedInside2_: bool
                                    d_25_appendedCurrent2_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out17_: bool
                                    out18_: _dafny.Seq
                                    out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next2_)
                                    d_23_appendedGenerated2_ = out16_
                                    d_24_appendedInside2_ = out17_
                                    d_25_appendedCurrent2_ = out18_
                                    generated = d_23_appendedGenerated2_
                                    insideConstrainedOut = d_24_appendedInside2_
                                    currentConstrainedOut = d_25_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

