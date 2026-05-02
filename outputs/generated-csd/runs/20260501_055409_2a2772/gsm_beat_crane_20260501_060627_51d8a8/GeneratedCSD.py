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
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        d_3_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_3_flatGroups_ = out0_
        d_4_tokensToPenalize_: _dafny.Seq
        d_4_tokensToPenalize_ = _dafny.SeqWithoutIsStrInference([])
        if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Let"))) in ((lm).Tokens):
            d_4_tokensToPenalize_ = (d_4_tokensToPenalize_) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Let"))]))
        if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "let"))) in ((lm).Tokens):
            d_4_tokensToPenalize_ = (d_4_tokensToPenalize_) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "let"))]))
        if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "P"))) in ((lm).Tokens):
            d_4_tokensToPenalize_ = (d_4_tokensToPenalize_) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "P"))]))
        if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore"))) in ((lm).Tokens):
            d_4_tokensToPenalize_ = (d_4_tokensToPenalize_) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Therefore"))]))
        if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "So"))) in ((lm).Tokens):
            d_4_tokensToPenalize_ = (d_4_tokensToPenalize_) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "So"))]))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        (lm).GenerateLogits((prompt) + (generated))
                        if ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) not in (generated)) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)):
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                            if (len(d_3_flatGroups_)) > (0):
                                (d_0_helpers_).BoostTokenLogits(lm, d_3_flatGroups_, _dafny.BigRational('3e0'))
                        elif (len(d_3_flatGroups_)) > (0):
                            (d_0_helpers_).PenalizeTokenLogits(lm, d_3_flatGroups_, _dafny.BigRational('1e0'))
                        d_5_next_: _dafny.Seq
                        out1_: _dafny.Seq
                        out1_ = (lm).ChooseNextTokenUnconstrained()
                        d_5_next_ = out1_
                        (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_5_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                            if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_6_closedGenerated_: _dafny.Seq
                            d_7_closedInside_: bool
                            d_8_closedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_6_closedGenerated_ = out2_
                            d_7_closedInside_ = out3_
                            d_8_closedCurrent_ = out4_
                            generated = d_6_closedGenerated_
                            insideConstrainedOut = d_7_closedInside_
                            currentConstrainedOut = d_8_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_stablePrefix_: _dafny.Seq
                            d_9_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_10_validCount_: int
                            out5_: int
                            out5_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_10_validCount_ = out5_
                            if ((len(d_4_tokensToPenalize_)) > (0)) and ((d_10_validCount_) > (d_2_narrowThreshold_)):
                                d_11_next_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = (d_0_helpers_).PenalizedConstrainedStep(lm, parser, (prompt) + (d_9_stablePrefix_), currentConstrainedOut, d_4_tokensToPenalize_, _dafny.BigRational('5e0'), eosToken)
                                d_11_next_ = out6_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_11_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_12_appendedGenerated_: _dafny.Seq
                                    d_13_appendedInside_: bool
                                    d_14_appendedCurrent_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                                    d_12_appendedGenerated_ = out7_
                                    d_13_appendedInside_ = out8_
                                    d_14_appendedCurrent_ = out9_
                                    generated = d_12_appendedGenerated_
                                    insideConstrainedOut = d_13_appendedInside_
                                    currentConstrainedOut = d_14_appendedCurrent_
                            elif True:
                                d_15_next_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_9_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                                d_15_next_ = out10_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_15_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_16_appendedGenerated_: _dafny.Seq
                                    d_17_appendedInside_: bool
                                    d_18_appendedCurrent_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                    d_16_appendedGenerated_ = out11_
                                    d_17_appendedInside_ = out12_
                                    d_18_appendedCurrent_ = out13_
                                    generated = d_16_appendedGenerated_
                                    insideConstrainedOut = d_17_appendedInside_
                                    currentConstrainedOut = d_18_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

