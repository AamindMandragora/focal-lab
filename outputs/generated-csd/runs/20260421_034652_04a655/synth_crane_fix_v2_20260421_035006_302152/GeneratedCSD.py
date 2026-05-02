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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, eosToken):
        generated: _dafny.Seq = _dafny.Seq({})
        insideConstrainedOut: bool = False
        currentConstrainedOut: _dafny.Seq = _dafny.Seq({})
        cost: int = int(0)
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__()
        d_0_helpers_ = nw0_
        generated = generatedPrefix
        insideConstrainedOut = False
        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_balance_: int
        d_2_balance_ = 0
        d_3_chunkLimit_: int
        d_3_chunkLimit_ = 4
        while (d_1_steps_) < (maxSteps):
            d_4_remaining_: int
            d_4_remaining_ = (maxSteps) - (d_1_steps_)
            if (d_4_remaining_) == (0):
                pass
            elif True:
                if (d_2_balance_) == (0):
                    (lm).GenerateLogits((prompt) + (generated))
                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('8e0'))
                    d_5_next0_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (lm).ChooseNextTokenUnconstrained()
                    d_5_next0_ = out0_
                    (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_5_next0_) == (eosToken):
                        pass
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next0_]))
                        if (d_5_next0_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_2_balance_ = (d_2_balance_) + (1)
                        elif True:
                            if ((d_5_next0_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))) and ((d_2_balance_) > (0)):
                                d_2_balance_ = (d_2_balance_) - (1)
                        if (d_1_steps_) < (maxSteps):
                            d_6_rem2_: int
                            d_6_rem2_ = (maxSteps) - (d_1_steps_)
                            d_7_useChunk_: int
                            d_7_useChunk_ = d_3_chunkLimit_
                            if (d_6_rem2_) < (d_7_useChunk_):
                                d_7_useChunk_ = d_6_rem2_
                            if (d_7_useChunk_) > (0):
                                d_8_gen2_: _dafny.Seq
                                d_9_stoppedOnOpenSpan_: bool
                                d_10_stoppedOnEos_: bool
                                d_11_stepsUsed_: int
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: bool
                                out4_: int
                                out1_, out2_, out3_, out4_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_7_useChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                                d_8_gen2_ = out1_
                                d_9_stoppedOnOpenSpan_ = out2_
                                d_10_stoppedOnEos_ = out3_
                                d_11_stepsUsed_ = out4_
                                generated = d_8_gen2_
                                d_1_steps_ = (d_1_steps_) + (d_11_stepsUsed_)
                                if d_9_stoppedOnOpenSpan_:
                                    d_2_balance_ = (d_2_balance_) + (1)
                                elif True:
                                    pass
                elif True:
                    (lm).GenerateLogits((prompt) + (generated))
                    (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('8e0'))
                    d_12_next1_: _dafny.Seq
                    out5_: _dafny.Seq
                    out5_ = (lm).ChooseNextTokenUnconstrained()
                    d_12_next1_ = out5_
                    (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_12_next1_) == (eosToken):
                        pass
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next1_]))
                        if (d_12_next1_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_2_balance_ = (d_2_balance_) + (1)
                        elif True:
                            if ((d_12_next1_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))) and ((d_2_balance_) > (0)):
                                d_2_balance_ = (d_2_balance_) - (1)
                        if (d_1_steps_) < (maxSteps):
                            d_13_rem3_: int
                            d_13_rem3_ = (maxSteps) - (d_1_steps_)
                            d_14_useChunk2_: int
                            d_14_useChunk2_ = d_3_chunkLimit_
                            if (d_13_rem3_) < (d_14_useChunk2_):
                                d_14_useChunk2_ = d_13_rem3_
                            if (d_14_useChunk2_) > (0):
                                d_15_gen3_: _dafny.Seq
                                d_16_stoppedOnOpenSpan2_: bool
                                d_17_stoppedOnEos2_: bool
                                d_18_stepsUsed2_: int
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: bool
                                out9_: int
                                out6_, out7_, out8_, out9_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_14_useChunk2_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                                d_15_gen3_ = out6_
                                d_16_stoppedOnOpenSpan2_ = out7_
                                d_17_stoppedOnEos2_ = out8_
                                d_18_stepsUsed2_ = out9_
                                generated = d_15_gen3_
                                d_1_steps_ = (d_1_steps_) + (d_18_stepsUsed2_)
                                if d_16_stoppedOnOpenSpan2_:
                                    d_2_balance_ = (d_2_balance_) + (1)
                                elif True:
                                    pass
            if (d_1_steps_) < (maxSteps):
                d_19_lastIsEosCheck_: bool
                d_19_lastIsEosCheck_ = False
            elif True:
                pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

