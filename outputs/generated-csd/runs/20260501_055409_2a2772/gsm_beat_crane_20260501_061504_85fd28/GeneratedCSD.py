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
        d_2_chunkSize_: int
        d_2_chunkSize_ = 6
        d_3_openedAny_: bool
        d_3_openedAny_ = (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in (generated)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_3_openedAny_)) and (((d_1_steps_) + (1)) < (maxSteps)):
                            (lm).GenerateLogits((prompt) + (generated))
                            if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens):
                                (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                            d_4_forced_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (lm).ChooseNextTokenUnconstrained()
                            d_4_forced_ = out0_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_4_forced_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                if (d_4_forced_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_forced_]))
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_3_openedAny_ = True
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_forced_]))
                        elif True:
                            d_5_remaining_: int
                            d_5_remaining_ = (maxSteps) - (d_1_steps_)
                            d_6_maxChunk_: int
                            d_6_maxChunk_ = d_2_chunkSize_
                            if (d_5_remaining_) < (d_6_maxChunk_):
                                d_6_maxChunk_ = d_5_remaining_
                            d_7_gChunk_: _dafny.Seq
                            d_8_stoppedOnOpenSpan_: bool
                            d_9_stoppedOnEos_: bool
                            d_10_stepsUsed_: int
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: bool
                            out4_: int
                            out1_, out2_, out3_, out4_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_6_maxChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_7_gChunk_ = out1_
                            d_8_stoppedOnOpenSpan_ = out2_
                            d_9_stoppedOnEos_ = out3_
                            d_10_stepsUsed_ = out4_
                            generated = d_7_gChunk_
                            d_1_steps_ = (d_1_steps_) + (d_10_stepsUsed_)
                            if d_9_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif True:
                                if d_8_stoppedOnOpenSpan_:
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_3_openedAny_ = True
                                elif True:
                                    if (d_1_steps_) < (maxSteps):
                                        (lm).GenerateLogits((prompt) + (generated))
                                        if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens):
                                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                                        d_11_next2_: _dafny.Seq
                                        out5_: _dafny.Seq
                                        out5_ = (lm).ChooseNextTokenUnconstrained()
                                        d_11_next2_ = out5_
                                        (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        if (d_11_next2_) == (eosToken):
                                            raise _dafny.Break("0")
                                        elif True:
                                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next2_]))
                                            if (d_11_next2_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                                insideConstrainedOut = True
                                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                                d_3_openedAny_ = True
                    elif True:
                        d_12_completeNow_: bool
                        d_12_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_12_completeNow_:
                            d_13_gClose_: _dafny.Seq
                            d_14_iClose_: bool
                            d_15_cClose_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_13_gClose_ = out6_
                            d_14_iClose_ = out7_
                            d_15_cClose_ = out8_
                            generated = d_13_gClose_
                            insideConstrainedOut = d_14_iClose_
                            currentConstrainedOut = d_15_cClose_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_16_narrow_: bool
                            out9_: bool
                            out9_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 2)
                            d_16_narrow_ = out9_
                            if d_16_narrow_:
                                d_17_repaired_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                d_17_repaired_ = out10_
                                generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - ((len(currentConstrainedOut)) - (len(d_17_repaired_))):])
                                currentConstrainedOut = d_17_repaired_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_18_stablePrefix_: _dafny.Seq
                                d_18_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_19_next3_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_18_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_19_next3_ = out11_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_19_next3_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_20_gApp_: _dafny.Seq
                                    d_21_iApp_: bool
                                    d_22_cApp_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next3_)
                                    d_20_gApp_ = out12_
                                    d_21_iApp_ = out13_
                                    d_22_cApp_ = out14_
                                    generated = d_20_gApp_
                                    insideConstrainedOut = d_21_iApp_
                                    currentConstrainedOut = d_22_cApp_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

