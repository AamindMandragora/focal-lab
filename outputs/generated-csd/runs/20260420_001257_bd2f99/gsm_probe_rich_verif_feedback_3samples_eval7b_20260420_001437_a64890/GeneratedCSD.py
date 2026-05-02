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
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_haveCompletedSpan_: bool
        d_2_haveCompletedSpan_ = False
        if not(insideConstrainedOut):
            d_3_idx_: int
            d_3_idx_ = 0
            while ((d_3_idx_) + (1)) < (len(generated)):
                if (((generated)[d_3_idx_]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and (((generated)[(d_3_idx_) + (1)]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))):
                    d_2_haveCompletedSpan_ = True
                    d_3_idx_ = (len(generated)) - (1)
                elif True:
                    d_3_idx_ = (d_3_idx_) + (1)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        d_4_completeNow_: bool
                        d_4_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_4_completeNow_:
                            d_5_gClose_: _dafny.Seq
                            d_6_iClose_: bool
                            d_7_cClose_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_5_gClose_ = out0_
                            d_6_iClose_ = out1_
                            d_7_cClose_ = out2_
                            generated = d_5_gClose_
                            insideConstrainedOut = d_6_iClose_
                            currentConstrainedOut = d_7_cClose_
                            d_2_haveCompletedSpan_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_stablePrefix_: _dafny.Seq
                            d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_9_constrainedPrompt_: _dafny.Seq
                            d_9_constrainedPrompt_ = (prompt) + (d_8_stablePrefix_)
                            d_10_validCount_: int
                            out3_: int
                            out3_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_10_validCount_ = out3_
                            if (d_10_validCount_) == (0):
                                d_11_gRoll_: _dafny.Seq
                                d_12_cRoll_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: _dafny.Seq
                                out4_, out5_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_8_stablePrefix_, generated, currentConstrainedOut)
                                d_11_gRoll_ = out4_
                                d_12_cRoll_ = out5_
                                generated = d_11_gRoll_
                                currentConstrainedOut = d_12_cRoll_
                                insideConstrainedOut = True
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_13_nextIn_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_13_nextIn_ = out6_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_13_nextIn_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_14_gApp_: _dafny.Seq
                                    d_15_iApp_: bool
                                    d_16_cApp_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_nextIn_)
                                    d_14_gApp_ = out7_
                                    d_15_iApp_ = out8_
                                    d_16_cApp_ = out9_
                                    generated = d_14_gApp_
                                    insideConstrainedOut = d_15_iApp_
                                    currentConstrainedOut = d_16_cApp_
                    elif True:
                        if not(d_2_haveCompletedSpan_):
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('1e2'))
                            d_17_nextOut_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (lm).ChooseNextTokenUnconstrained()
                            d_17_nextOut_ = out10_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_17_nextOut_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                if (d_17_nextOut_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_18_gOpen_: _dafny.Seq
                                    d_19_iOpen_: bool
                                    d_20_cOpen_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_18_gOpen_ = out11_
                                    d_19_iOpen_ = out12_
                                    d_20_cOpen_ = out13_
                                    generated = d_18_gOpen_
                                    insideConstrainedOut = d_19_iOpen_
                                    currentConstrainedOut = d_20_cOpen_
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_17_nextOut_]))
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('3e0'))
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e0'))
                            d_21_nextFree_: _dafny.Seq
                            out14_: _dafny.Seq
                            out14_ = (lm).ChooseNextTokenUnconstrained()
                            d_21_nextFree_ = out14_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_21_nextFree_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                if (d_21_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_22_gOpen2_: _dafny.Seq
                                    d_23_iOpen2_: bool
                                    d_24_cOpen2_: _dafny.Seq
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: _dafny.Seq
                                    out15_, out16_, out17_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_22_gOpen2_ = out15_
                                    d_23_iOpen2_ = out16_
                                    d_24_cOpen2_ = out17_
                                    generated = d_22_gOpen2_
                                    insideConstrainedOut = d_23_iOpen2_
                                    currentConstrainedOut = d_24_cOpen2_
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_21_nextFree_]))
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

