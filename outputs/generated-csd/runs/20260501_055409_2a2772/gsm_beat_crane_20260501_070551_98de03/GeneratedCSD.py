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
        d_2_openedSpan_: bool
        d_2_openedSpan_ = insideConstrained
        d_3_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_3_flatGroups_ = out0_
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
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_5_gClose_ = out1_
                            d_6_iClose_ = out2_
                            d_7_cClose_ = out3_
                            generated = d_5_gClose_
                            insideConstrainedOut = d_6_iClose_
                            currentConstrainedOut = d_7_cClose_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            if ((len(currentConstrainedOut)) >= (stepTokenBudget)) or ((len(currentConstrainedOut)) >= (24)):
                                d_8_repaired_: _dafny.Seq
                                out4_: _dafny.Seq
                                out4_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                d_8_repaired_ = out4_
                                generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - ((len(currentConstrainedOut)) - (len(d_8_repaired_))):])
                                currentConstrainedOut = d_8_repaired_
                                d_9_repairedComplete_: bool
                                d_9_repairedComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if d_9_repairedComplete_:
                                    d_10_gClose2_: _dafny.Seq
                                    d_11_iClose2_: bool
                                    d_12_cClose2_: _dafny.Seq
                                    out5_: _dafny.Seq
                                    out6_: bool
                                    out7_: _dafny.Seq
                                    out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_10_gClose2_ = out5_
                                    d_11_iClose2_ = out6_
                                    d_12_cClose2_ = out7_
                                    generated = d_10_gClose2_
                                    insideConstrainedOut = d_11_iClose2_
                                    currentConstrainedOut = d_12_cClose2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    raise _dafny.Break("0")
                                elif True:
                                    d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_13_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                                if (len(d_3_flatGroups_)) > (0):
                                    out8_: _dafny.Seq
                                    out8_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (generated), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                    d_13_next_ = out8_
                                elif True:
                                    out9_: _dafny.Seq
                                    out9_ = (d_0_helpers_).PenalizedConstrainedStep(lm, parser, (prompt) + (generated), currentConstrainedOut, d_3_flatGroups_, _dafny.BigRational('0e0'), eosToken)
                                    d_13_next_ = out9_
                                if (d_13_next_) == (eosToken):
                                    d_14_repaired2_: _dafny.Seq
                                    out10_: _dafny.Seq
                                    out10_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                                    d_14_repaired2_ = out10_
                                    generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - ((len(currentConstrainedOut)) - (len(d_14_repaired2_))):])
                                    currentConstrainedOut = d_14_repaired2_
                                    d_15_repaired2Complete_: bool
                                    d_15_repaired2Complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                    if d_15_repaired2Complete_:
                                        d_16_gClose3_: _dafny.Seq
                                        d_17_iClose3_: bool
                                        d_18_cClose3_: _dafny.Seq
                                        out11_: _dafny.Seq
                                        out12_: bool
                                        out13_: _dafny.Seq
                                        out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_16_gClose3_ = out11_
                                        d_17_iClose3_ = out12_
                                        d_18_cClose3_ = out13_
                                        generated = d_16_gClose3_
                                        insideConstrainedOut = d_17_iClose3_
                                        currentConstrainedOut = d_18_cClose3_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        insideConstrainedOut = False
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    raise _dafny.Break("0")
                                elif True:
                                    d_19_gApp_: _dafny.Seq
                                    d_20_iApp_: bool
                                    d_21_cApp_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                                    d_19_gApp_ = out14_
                                    d_20_iApp_ = out15_
                                    d_21_cApp_ = out16_
                                    generated = d_19_gApp_
                                    insideConstrainedOut = d_20_iApp_
                                    currentConstrainedOut = d_21_cApp_
                                    d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        if (not(d_2_openedSpan_)) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)):
                            d_22_gOpen_: _dafny.Seq
                            d_23_iOpen_: bool
                            d_24_cOpen_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: bool
                            out19_: _dafny.Seq
                            out17_, out18_, out19_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_22_gOpen_ = out17_
                            d_23_iOpen_ = out18_
                            d_24_cOpen_ = out19_
                            generated = d_22_gOpen_
                            insideConstrainedOut = d_23_iOpen_
                            currentConstrainedOut = d_24_cOpen_
                            d_2_openedSpan_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_25_nextFree_: _dafny.Seq
                            out20_: _dafny.Seq
                            out20_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_25_nextFree_ = out20_
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_25_nextFree_]))
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

