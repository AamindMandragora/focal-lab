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
        d_2_shouldOpen_: bool
        d_2_shouldOpen_ = False
        d_3_stablePrefix_: _dafny.Seq
        d_3_stablePrefix_ = _dafny.SeqWithoutIsStrInference([])
        d_4_next_: _dafny.Seq
        d_4_next_ = eosToken
        d_5_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_5_flatGroups_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_shouldOpen_ = False
                        if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens):
                            if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))) in (generated):
                                d_2_shouldOpen_ = True
                            elif True:
                                if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+"))) in (generated):
                                    d_2_shouldOpen_ = True
                                elif True:
                                    if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-"))) in (generated):
                                        d_2_shouldOpen_ = True
                                    elif True:
                                        if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*"))) in (generated):
                                            d_2_shouldOpen_ = True
                                        elif True:
                                            if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/"))) in (generated):
                                                d_2_shouldOpen_ = True
                                            elif True:
                                                if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total"))) in (generated):
                                                    d_2_shouldOpen_ = True
                        if d_2_shouldOpen_:
                            d_6_openedGenerated_: _dafny.Seq
                            d_7_openedInside_: bool
                            d_8_openedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_openedGenerated_ = out1_
                            d_7_openedInside_ = out2_
                            d_8_openedCurrent_ = out3_
                            generated = d_6_openedGenerated_
                            insideConstrainedOut = d_7_openedInside_
                            currentConstrainedOut = d_8_openedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens):
                                (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('8e0'))
                            if (len(d_5_flatGroups_)) > (0):
                                (d_0_helpers_).PenalizeTokenLogits(lm, d_5_flatGroups_, _dafny.BigRational('1e0'))
                            out4_: _dafny.Seq
                            out4_ = (lm).ChooseNextTokenUnconstrained()
                            d_4_next_ = out4_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_4_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                                if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
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
                            d_3_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            if (len(validTokenGroups)) > (0):
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, (prompt) + (d_3_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                                d_4_next_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_4_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_12_appendedGenerated_: _dafny.Seq
                                    d_13_appendedInside_: bool
                                    d_14_appendedCurrent_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_4_next_)
                                    d_12_appendedGenerated_ = out9_
                                    d_13_appendedInside_ = out10_
                                    d_14_appendedCurrent_ = out11_
                                    generated = d_12_appendedGenerated_
                                    insideConstrainedOut = d_13_appendedInside_
                                    currentConstrainedOut = d_14_appendedCurrent_
                            elif True:
                                d_15_generatedOut_: _dafny.Seq
                                d_16_insideOut_: bool
                                d_17_currentOut_: _dafny.Seq
                                d_18_hitEos_: bool
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out15_: bool
                                out12_, out13_, out14_, out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, (prompt) + (d_3_stablePrefix_), generated, currentConstrainedOut, eosToken)
                                d_15_generatedOut_ = out12_
                                d_16_insideOut_ = out13_
                                d_17_currentOut_ = out14_
                                d_18_hitEos_ = out15_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if d_18_hitEos_:
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = d_15_generatedOut_
                                    insideConstrainedOut = d_16_insideOut_
                                    currentConstrainedOut = d_17_currentOut_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

