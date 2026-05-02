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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_2_completeNow_: bool
                        d_2_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_2_completeNow_:
                            d_3_genClosed_: _dafny.Seq
                            d_4_insideClosed_: bool
                            d_5_curClosed_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_3_genClosed_ = out0_
                            d_4_insideClosed_ = out1_
                            d_5_curClosed_ = out2_
                            generated = d_3_genClosed_
                            insideConstrainedOut = d_4_insideClosed_
                            currentConstrainedOut = d_5_curClosed_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_6_remaining1_: int
                            d_6_remaining1_ = (maxSteps) - (d_1_steps_)
                            if (d_6_remaining1_) == (0):
                                raise _dafny.Break("0")
                            elif True:
                                d_7_curNext_: _dafny.Seq
                                d_8_hitEos_: bool
                                d_9_stepsUsed_: int
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: int
                                out3_, out4_, out5_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])), currentConstrainedOut, d_6_remaining1_, eosToken)
                                d_7_curNext_ = out3_
                                d_8_hitEos_ = out4_
                                d_9_stepsUsed_ = out5_
                                if (d_9_stepsUsed_) == (0):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])) + (d_7_curNext_)
                                    currentConstrainedOut = d_7_curNext_
                                    d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                                    if d_8_hitEos_:
                                        raise _dafny.Break("0")
                    elif True:
                        d_10_next_: _dafny.Seq
                        out6_: _dafny.Seq
                        out6_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_10_next_ = out6_
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_10_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            if (VerifiedDecoderAgent.default__.Contains(d_10_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and ((d_1_steps_) < (maxSteps)):
                                d_11_genOpen_: _dafny.Seq
                                d_12_insideOpen_: bool
                                d_13_curOpen_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_11_genOpen_ = out7_
                                d_12_insideOpen_ = out8_
                                d_13_curOpen_ = out9_
                                generated = d_11_genOpen_
                                insideConstrainedOut = d_12_insideOpen_
                                currentConstrainedOut = d_13_curOpen_
                                d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

