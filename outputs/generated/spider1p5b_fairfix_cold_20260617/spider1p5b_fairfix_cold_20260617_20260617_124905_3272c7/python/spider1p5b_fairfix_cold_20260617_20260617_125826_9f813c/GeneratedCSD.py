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
        if True:
            generated = generatedPrefix
            insideConstrainedOut = insideConstrained
            currentConstrainedOut = currentConstrained
            cost = 0
            (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate valid SQL for the spider dataset. Use hard parser-constrained decoding.")))
            d_1_steps_: int
            d_1_steps_ = 0
            with _dafny.label("0"):
                while (d_1_steps_) < (maxSteps):
                    with _dafny.c_label("0"):
                        if insideConstrainedOut:
                            d_2_cg_: _dafny.Seq
                            d_3_ci_: bool
                            d_4_cc_: _dafny.Seq
                            d_5_closed_: bool
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out3_: bool
                            out0_, out1_, out2_, out3_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_2_cg_ = out0_
                            d_3_ci_ = out1_
                            d_4_cc_ = out2_
                            d_5_closed_ = out3_
                            if d_5_closed_:
                                generated = d_2_cg_
                                insideConstrainedOut = d_3_ci_
                                currentConstrainedOut = d_4_cc_
                                d_1_steps_ = (d_1_steps_) + (1)
                                cost = d_1_steps_
                                raise _dafny.Break("0")
                            d_6_next_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, prompt, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_6_next_ = out4_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_6_next_) == (eosToken):
                                cost = d_1_steps_
                                raise _dafny.Break("0")
                            d_7_ng_: _dafny.Seq
                            d_8_ni_: bool
                            d_9_nc_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_6_next_)
                            d_7_ng_ = out5_
                            d_8_ni_ = out6_
                            d_9_nc_ = out7_
                            generated = d_7_ng_
                            insideConstrainedOut = d_8_ni_
                            currentConstrainedOut = d_9_nc_
                        elif True:
                            d_10_next_: _dafny.Seq
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_10_next_ = out8_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_10_next_) == (eosToken):
                                cost = d_1_steps_
                                raise _dafny.Break("0")
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                            if (d_10_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_11_og_: _dafny.Seq
                                d_12_oi_: bool
                                d_13_oc_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_11_og_ = out9_
                                d_12_oi_ = out10_
                                d_13_oc_ = out11_
                                generated = d_11_og_
                                insideConstrainedOut = d_12_oi_
                                currentConstrainedOut = d_13_oc_
                        cost = d_1_steps_
                        pass
                pass
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

