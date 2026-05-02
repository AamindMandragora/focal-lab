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
        d_2_openedOnce_: bool
        d_2_openedOnce_ = insideConstrained
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_nextOutside_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_3_nextOutside_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_3_nextOutside_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_nextOutside_]))
                            if (not(d_2_openedOnce_)) and (VerifiedDecoderAgent.default__.Contains(d_3_nextOutside_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_openedOnce_ = True
                    elif True:
                        d_4_complete_: bool
                        d_4_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_4_complete_:
                            if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))) in ((lm).Tokens):
                                d_5_closedGenerated_: _dafny.Seq
                                d_6_closedInside_: bool
                                d_7_closedCurrent_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_5_closedGenerated_ = out1_
                                d_6_closedInside_ = out2_
                                d_7_closedCurrent_ = out3_
                                generated = d_5_closedGenerated_
                                insideConstrainedOut = d_6_closedInside_
                                currentConstrainedOut = d_7_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            if (10) <= (len(currentConstrainedOut)):
                                d_8_oldLen_: int
                                d_8_oldLen_ = len(currentConstrainedOut)
                                d_9_repaired_: _dafny.Seq
                                d_10_excludedTok_: _dafny.Seq
                                d_11_hasExcluded_: bool
                                out4_: _dafny.Seq
                                out5_: _dafny.Seq
                                out6_: bool
                                out4_, out5_, out6_ = VerifiedDecoderAgent.CSDHelpers.RollbackAndExclude(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_9_repaired_ = out4_
                                d_10_excludedTok_ = out5_
                                d_11_hasExcluded_ = out6_
                                currentConstrainedOut = d_9_repaired_
                                generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - ((d_8_oldLen_) - (len(d_9_repaired_))):])
                                d_12_stablePrefixRecovered_: _dafny.Seq
                                d_12_stablePrefixRecovered_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                if (d_11_hasExcluded_) and ((d_10_excludedTok_) in ((lm).Tokens)):
                                    d_13_penalizedNext_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out7_ = (d_0_helpers_).PenalizedConstrainedStep(lm, parser, (prompt) + (d_12_stablePrefixRecovered_), currentConstrainedOut, _dafny.SeqWithoutIsStrInference([d_10_excludedTok_]), _dafny.BigRational('5e0'), eosToken)
                                    d_13_penalizedNext_ = out7_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_13_penalizedNext_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_14_appendedGenerated1_: _dafny.Seq
                                        d_15_appendedInside1_: bool
                                        d_16_appendedCurrent1_: _dafny.Seq
                                        out8_: _dafny.Seq
                                        out9_: bool
                                        out10_: _dafny.Seq
                                        out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_penalizedNext_)
                                        d_14_appendedGenerated1_ = out8_
                                        d_15_appendedInside1_ = out9_
                                        d_16_appendedCurrent1_ = out10_
                                        generated = d_14_appendedGenerated1_
                                        insideConstrainedOut = d_15_appendedInside1_
                                        currentConstrainedOut = d_16_appendedCurrent1_
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                d_17_stablePrefix_: _dafny.Seq
                                d_17_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_18_nextInside_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).PenalizedConstrainedStep(lm, parser, (prompt) + (d_17_stablePrefix_), currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('3e0'), eosToken)
                                d_18_nextInside_ = out11_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_18_nextInside_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_19_appendedGenerated_: _dafny.Seq
                                    d_20_appendedInside_: bool
                                    d_21_appendedCurrent_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_nextInside_)
                                    d_19_appendedGenerated_ = out12_
                                    d_20_appendedInside_ = out13_
                                    d_21_appendedCurrent_ = out14_
                                    generated = d_19_appendedGenerated_
                                    insideConstrainedOut = d_20_appendedInside_
                                    currentConstrainedOut = d_21_appendedCurrent_
                    pass
            pass
        if (((((maxSteps) > (0)) and ((d_1_steps_) == (0))) and ((generated) == (generatedPrefix))) and ((insideConstrainedOut) == (insideConstrained))) and ((currentConstrainedOut) == (currentConstrained)):
            if insideConstrainedOut:
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            elif True:
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([eosToken]))
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

