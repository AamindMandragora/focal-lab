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
        d_2_eqToken_: _dafny.Seq
        d_2_eqToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))
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
                    elif True:
                        d_4_complete_: bool
                        d_4_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_4_complete_:
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
                            d_8_stablePrefix_: _dafny.Seq
                            d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_9_prevTok_: _dafny.Seq
                            d_10_hasEq_: bool
                            out4_: _dafny.Seq
                            out5_: bool
                            out4_, out5_ = (d_0_helpers_).LastTokenBefore((currentConstrainedOut) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))])), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                            d_9_prevTok_ = out4_
                            d_10_hasEq_ = out5_
                            if ((d_10_hasEq_) and ((d_9_prevTok_) == (d_2_eqToken_))) and ((d_2_eqToken_) in ((lm).Tokens)):
                                d_11_nextPen_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = (d_0_helpers_).PenalizedConstrainedStep(lm, parser, (prompt) + (d_8_stablePrefix_), currentConstrainedOut, _dafny.SeqWithoutIsStrInference([d_2_eqToken_]), _dafny.BigRational('5e0'), eosToken)
                                d_11_nextPen_ = out6_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_11_nextPen_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_12_appendedGenerated1_: _dafny.Seq
                                    d_13_appendedInside1_: bool
                                    d_14_appendedCurrent1_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_nextPen_)
                                    d_12_appendedGenerated1_ = out7_
                                    d_13_appendedInside1_ = out8_
                                    d_14_appendedCurrent1_ = out9_
                                    generated = d_12_appendedGenerated1_
                                    insideConstrainedOut = d_13_appendedInside1_
                                    currentConstrainedOut = d_14_appendedCurrent1_
                            elif True:
                                d_15_nextAdaptive_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_8_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_15_nextAdaptive_ = out10_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_15_nextAdaptive_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_16_appendedGenerated2_: _dafny.Seq
                                    d_17_appendedInside2_: bool
                                    d_18_appendedCurrent2_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_nextAdaptive_)
                                    d_16_appendedGenerated2_ = out11_
                                    d_17_appendedInside2_ = out12_
                                    d_18_appendedCurrent2_ = out13_
                                    generated = d_16_appendedGenerated2_
                                    insideConstrainedOut = d_17_appendedInside2_
                                    currentConstrainedOut = d_18_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

