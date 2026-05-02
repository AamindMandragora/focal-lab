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
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_openedHere_ = True
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                    elif True:
                        d_3_completeNow_: bool
                        d_3_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_3_completeNow_:
                            if ((d_1_steps_) + (1)) <= (maxSteps):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_4_dead_: bool
                            out0_: bool
                            out0_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_4_dead_ = out0_
                            if d_4_dead_:
                                d_5_stableDead_: _dafny.Seq
                                d_5_stableDead_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_6_gRoll_: _dafny.Seq
                                d_7_cRoll_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: _dafny.Seq
                                out1_, out2_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_5_stableDead_, generated, currentConstrainedOut)
                                d_6_gRoll_ = out1_
                                d_7_cRoll_ = out2_
                                generated = d_6_gRoll_
                                currentConstrainedOut = d_7_cRoll_
                                insideConstrainedOut = True
                            elif True:
                                if (stepTokenBudget) == (0):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_8_remaining_: int
                                    d_8_remaining_ = (maxSteps) - (d_1_steps_)
                                    d_9_localBudget_: int = int(0)
                                    if (stepTokenBudget) <= (d_8_remaining_):
                                        d_9_localBudget_ = stepTokenBudget
                                    elif True:
                                        d_9_localBudget_ = d_8_remaining_
                                    d_10_stable_: _dafny.Seq
                                    d_10_stable_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    d_11_constrainedPrompt_: _dafny.Seq
                                    d_11_constrainedPrompt_ = (prompt) + (d_10_stable_)
                                    d_12_currentNew_: _dafny.Seq
                                    d_13_hitEos_: bool
                                    d_14_used_: int
                                    out3_: _dafny.Seq
                                    out4_: bool
                                    out5_: int
                                    out3_, out4_, out5_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, d_9_localBudget_, eosToken)
                                    d_12_currentNew_ = out3_
                                    d_13_hitEos_ = out4_
                                    d_14_used_ = out5_
                                    if (d_13_hitEos_) or ((d_14_used_) == (0)):
                                        raise _dafny.Break("0")
                                    elif True:
                                        if ((d_1_steps_) + (d_14_used_)) <= (maxSteps):
                                            generated = (d_10_stable_) + (d_12_currentNew_)
                                            currentConstrainedOut = d_12_currentNew_
                                            insideConstrainedOut = True
                                            d_1_steps_ = (d_1_steps_) + (d_14_used_)
                                        elif True:
                                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

