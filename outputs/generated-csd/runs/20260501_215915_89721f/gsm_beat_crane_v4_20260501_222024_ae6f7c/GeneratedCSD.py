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
        d_2_useGroups_: bool
        d_2_useGroups_ = (0) < (len(validTokenGroups))
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
                            if ((d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or (VerifiedDecoderAgent.default__.Contains(d_3_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))):
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
                            d_9_nextConstrained_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            if d_2_useGroups_:
                                out4_: _dafny.Seq
                                out4_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_8_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_9_nextConstrained_ = out4_
                            elif True:
                                d_10_g2_: _dafny.Seq
                                d_11_i2_: bool
                                d_12_c2_: _dafny.Seq
                                d_13_hitEos_: bool
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: _dafny.Seq
                                out8_: bool
                                out5_, out6_, out7_, out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, (prompt) + (d_8_stablePrefix_), generated, currentConstrainedOut, eosToken)
                                d_10_g2_ = out5_
                                d_11_i2_ = out6_
                                d_12_c2_ = out7_
                                d_13_hitEos_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if d_13_hitEos_:
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = d_10_g2_
                                    insideConstrainedOut = d_11_i2_
                                    currentConstrainedOut = d_12_c2_
                                d_9_nextConstrained_ = eosToken
                            if d_2_useGroups_:
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_9_nextConstrained_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_14_appendedGenerated_: _dafny.Seq
                                    d_15_appendedInside_: bool
                                    d_16_appendedCurrent_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_nextConstrained_)
                                    d_14_appendedGenerated_ = out9_
                                    d_15_appendedInside_ = out10_
                                    d_16_appendedCurrent_ = out11_
                                    generated = d_14_appendedGenerated_
                                    insideConstrainedOut = d_15_appendedInside_
                                    currentConstrainedOut = d_16_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

