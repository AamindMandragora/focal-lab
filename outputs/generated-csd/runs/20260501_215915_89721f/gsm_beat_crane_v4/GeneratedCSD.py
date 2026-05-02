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
        d_2_boundaryToken_: _dafny.Seq
        d_2_boundaryToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ","))
        d_3_narrowThreshold_: int
        d_3_narrowThreshold_ = 2
        d_4_penaltyTokens_: _dafny.Seq
        d_4_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_5_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_5_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                            if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_6_isComplete_: bool
                        d_6_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_6_isComplete_:
                            d_7_closedGenerated_: _dafny.Seq
                            d_8_closedInside_: bool
                            d_9_closedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_7_closedGenerated_ = out1_
                            d_8_closedInside_ = out2_
                            d_9_closedCurrent_ = out3_
                            generated = d_7_closedGenerated_
                            insideConstrainedOut = d_8_closedInside_
                            currentConstrainedOut = d_9_closedCurrent_
                            d_4_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([])
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_10_stablePrefix_: _dafny.Seq
                            d_10_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_11_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            def lambda0_(forall_var_0_):
                                d_12_t_: _dafny.Seq = forall_var_0_
                                return not ((d_12_t_) in (d_4_penaltyTokens_)) or ((d_12_t_) in ((lm).Tokens))

                            if ((0) < (len(d_4_penaltyTokens_))) and (_dafny.quantifier((d_4_penaltyTokens_).UniqueElements, True, lambda0_)):
                                out4_: _dafny.Seq
                                out4_ = (d_0_helpers_).PenalizedConstrainedStep(lm, parser, (prompt) + (d_10_stablePrefix_), currentConstrainedOut, d_4_penaltyTokens_, _dafny.BigRational('5e0'), eosToken)
                                d_11_next_ = out4_
                                d_4_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([])
                            elif True:
                                out5_: _dafny.Seq
                                out5_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_10_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_11_next_ = out5_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_11_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_13_appendedGenerated_: _dafny.Seq
                                d_14_appendedInside_: bool
                                d_15_appendedCurrent_: _dafny.Seq
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: _dafny.Seq
                                out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                                d_13_appendedGenerated_ = out6_
                                d_14_appendedInside_ = out7_
                                d_15_appendedCurrent_ = out8_
                                generated = d_13_appendedGenerated_
                                insideConstrainedOut = d_14_appendedInside_
                                currentConstrainedOut = d_15_appendedCurrent_
                                d_16_validCount_: int
                                out9_: int
                                out9_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                d_16_validCount_ = out9_
                                if ((d_16_validCount_) < (d_3_narrowThreshold_)) and ((len(currentConstrainedOut)) > (0)):
                                    d_17_repaired_: _dafny.Seq
                                    d_18_excludedTok_: _dafny.Seq
                                    d_19_hasExcluded_: bool
                                    out10_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out10_, out11_, out12_ = VerifiedDecoderAgent.CSDHelpers.RollbackAndExclude(parser, currentConstrainedOut, d_2_boundaryToken_)
                                    d_17_repaired_ = out10_
                                    d_18_excludedTok_ = out11_
                                    d_19_hasExcluded_ = out12_
                                    d_20_dropped_: int
                                    d_20_dropped_ = (len(currentConstrainedOut)) - (len(d_17_repaired_))
                                    generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (d_20_dropped_):])
                                    currentConstrainedOut = d_17_repaired_
                                    if (d_19_hasExcluded_) and ((d_18_excludedTok_) in ((lm).Tokens)):
                                        d_4_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([d_18_excludedTok_])
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

