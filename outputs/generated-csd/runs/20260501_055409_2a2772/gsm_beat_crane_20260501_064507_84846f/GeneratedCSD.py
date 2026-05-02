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
        d_2_usedPrelude_: bool
        d_2_usedPrelude_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_3_completeNow_: bool
                        d_3_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_3_completeNow_:
                            if ((d_1_steps_) + (1)) <= (maxSteps):
                                d_4_gClose_: _dafny.Seq
                                d_5_iClose_: bool
                                d_6_cClose_: _dafny.Seq
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: _dafny.Seq
                                out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_4_gClose_ = out0_
                                d_5_iClose_ = out1_
                                d_6_cClose_ = out2_
                                generated = d_4_gClose_
                                insideConstrainedOut = d_5_iClose_
                                currentConstrainedOut = d_6_cClose_
                                d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            if ((d_1_steps_) + (1)) <= (maxSteps):
                                d_7_next_: _dafny.Seq
                                out3_: _dafny.Seq
                                out3_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, (prompt) + (generated), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                                d_7_next_ = out3_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_7_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_8_gApp_: _dafny.Seq
                                    d_9_iApp_: bool
                                    d_10_cApp_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_7_next_)
                                    d_8_gApp_ = out4_
                                    d_9_iApp_ = out5_
                                    d_10_cApp_ = out6_
                                    generated = d_8_gApp_
                                    insideConstrainedOut = d_9_iApp_
                                    currentConstrainedOut = d_10_cApp_
                            elif True:
                                raise _dafny.Break("0")
                    elif True:
                        if (((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)) and (not(d_2_usedPrelude_))) and (((d_1_steps_) + (2)) <= (maxSteps)):
                            d_11_nextUn_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_11_nextUn_ = out7_
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_nextUn_]))
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_usedPrelude_ = True
                            if (d_11_nextUn_) == (eosToken):
                                raise _dafny.Break("0")
                        elif True:
                            if ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                d_12_gOpen_: _dafny.Seq
                                d_13_iOpen_: bool
                                d_14_cOpen_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_12_gOpen_ = out8_
                                d_13_iOpen_ = out9_
                                d_14_cOpen_ = out10_
                                generated = d_12_gOpen_
                                insideConstrainedOut = d_13_iOpen_
                                currentConstrainedOut = d_14_cOpen_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                if ((d_1_steps_) + (1)) <= (maxSteps):
                                    d_15_nextFree_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out11_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                    d_15_nextFree_ = out11_
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_15_nextFree_]))
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_15_nextFree_) == (eosToken):
                                        raise _dafny.Break("0")
                                elif True:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

