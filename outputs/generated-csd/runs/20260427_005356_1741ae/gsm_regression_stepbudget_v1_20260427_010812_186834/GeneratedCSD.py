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
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        (lm).GenerateLogits((prompt) + (generated))
                        (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                        d_2_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (lm).ChooseNextTokenUnconstrained()
                        d_2_next_ = out0_
                        (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_2_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            if (d_2_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_3_isComplete_: bool
                        d_3_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_3_isComplete_:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('1e2'))
                            d_4_next2_: _dafny.Seq
                            out1_: _dafny.Seq
                            out1_ = (lm).ChooseNextToken()
                            d_4_next2_ = out1_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next2_]))
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_4_next2_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                if (d_4_next2_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                elif True:
                                    raise _dafny.Break("0")
                        elif True:
                            d_5_stablePrefix_: _dafny.Seq
                            d_5_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_6_next3_: _dafny.Seq
                            out2_: _dafny.Seq
                            out2_ = (d_0_helpers_).ConstrainedStep(lm, parser, (prompt) + (d_5_stablePrefix_), currentConstrainedOut, eosToken)
                            d_6_next3_ = out2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_6_next3_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_7_appendedGenerated2_: _dafny.Seq
                                d_8_appendedInside2_: bool
                                d_9_appendedCurrent2_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_6_next3_)
                                d_7_appendedGenerated2_ = out3_
                                d_8_appendedInside2_ = out4_
                                d_9_appendedCurrent2_ = out5_
                                generated = d_7_appendedGenerated2_
                                insideConstrainedOut = d_8_appendedInside2_
                                currentConstrainedOut = d_9_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        if (maxSteps) > (0):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

