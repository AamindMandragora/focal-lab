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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, eosToken):
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
        d_2_openedHere_: bool
        d_2_openedHere_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_2_openedHere_:
                            raise _dafny.Break("0")
                        elif True:
                            if ((d_1_steps_) + (1)) <= (maxSteps):
                                d_3_gOpen_: _dafny.Seq
                                d_4_inOpen_: bool
                                d_5_cOpen_: _dafny.Seq
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: _dafny.Seq
                                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_3_gOpen_ = out0_
                                d_4_inOpen_ = out1_
                                d_5_cOpen_ = out2_
                                generated = d_3_gOpen_
                                insideConstrainedOut = d_4_inOpen_
                                currentConstrainedOut = d_5_cOpen_
                                d_2_openedHere_ = True
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                    elif True:
                        d_6_completeNow_: bool
                        d_6_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_6_completeNow_:
                            if ((d_1_steps_) + (1)) <= (maxSteps):
                                d_7_gClose_: _dafny.Seq
                                d_8_inClose_: bool
                                d_9_cClose_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_7_gClose_ = out3_
                                d_8_inClose_ = out4_
                                d_9_cClose_ = out5_
                                generated = d_7_gClose_
                                insideConstrainedOut = d_8_inClose_
                                currentConstrainedOut = d_9_cClose_
                                d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_10_dead_: bool
                            out6_: bool
                            out6_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_10_dead_ = out6_
                            if d_10_dead_:
                                d_11_stableDead_: _dafny.Seq
                                d_11_stableDead_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_12_gRoll_: _dafny.Seq
                                d_13_cRoll_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: _dafny.Seq
                                out7_, out8_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_11_stableDead_, generated, currentConstrainedOut)
                                d_12_gRoll_ = out7_
                                d_13_cRoll_ = out8_
                                generated = d_12_gRoll_
                                currentConstrainedOut = d_13_cRoll_
                                insideConstrainedOut = True
                            elif True:
                                if (stepTokenBudget) == (0):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_14_remaining_: int
                                    d_14_remaining_ = (maxSteps) - (d_1_steps_)
                                    d_15_localBudget_: int = int(0)
                                    if (stepTokenBudget) <= (d_14_remaining_):
                                        d_15_localBudget_ = stepTokenBudget
                                    elif True:
                                        d_15_localBudget_ = d_14_remaining_
                                    d_16_stable_: _dafny.Seq
                                    d_16_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    d_17_constrainedPrompt_: _dafny.Seq
                                    d_17_constrainedPrompt_ = (prompt) + (d_16_stable_)
                                    d_18_currentNew_: _dafny.Seq
                                    d_19_hitEos_: bool
                                    d_20_used_: int
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: int
                                    out9_, out10_, out11_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, d_15_localBudget_, eosToken)
                                    d_18_currentNew_ = out9_
                                    d_19_hitEos_ = out10_
                                    d_20_used_ = out11_
                                    if (d_19_hitEos_) or ((d_20_used_) == (0)):
                                        raise _dafny.Break("0")
                                    elif True:
                                        if ((d_1_steps_) + (d_20_used_)) <= (maxSteps):
                                            generated = (d_16_stable_) + (d_18_currentNew_)
                                            currentConstrainedOut = d_18_currentNew_
                                            insideConstrainedOut = True
                                            d_1_steps_ = (d_1_steps_) + (d_20_used_)
                                        elif True:
                                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

