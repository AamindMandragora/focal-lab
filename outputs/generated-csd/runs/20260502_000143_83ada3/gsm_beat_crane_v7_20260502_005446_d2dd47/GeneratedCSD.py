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
        d_2_spansClosed_: int
        d_2_spansClosed_ = 0
        d_3_openWarmup1_: int
        d_3_openWarmup1_ = 5
        d_4_openWarmup2_: int
        d_4_openWarmup2_ = 12
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_2_spansClosed_) > (0):
                            raise _dafny.Break("0")
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens):
                                if (d_4_openWarmup2_) <= (d_1_steps_):
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('14e0'))
                                elif (d_3_openWarmup1_) <= (d_1_steps_):
                                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('6e0'))
                            d_5_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (lm).ChooseNextTokenUnconstrained()
                            d_5_next_ = out0_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
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
                            d_2_spansClosed_ = (d_2_spansClosed_) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_10_stablePrefix_: _dafny.Seq
                            d_10_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_11_constrainedGenerated_: _dafny.Seq
                            d_12_constrainedInside_: bool
                            d_13_constrainedCurrent_: _dafny.Seq
                            d_14_hitEos_: bool
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out7_: bool
                            out4_, out5_, out6_, out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, (prompt) + (d_10_stablePrefix_), generated, currentConstrainedOut, eosToken)
                            d_11_constrainedGenerated_ = out4_
                            d_12_constrainedInside_ = out5_
                            d_13_constrainedCurrent_ = out6_
                            d_14_hitEos_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if d_14_hitEos_:
                                raise _dafny.Break("0")
                            elif True:
                                generated = d_11_constrainedGenerated_
                                insideConstrainedOut = d_12_constrainedInside_
                                currentConstrainedOut = d_13_constrainedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

