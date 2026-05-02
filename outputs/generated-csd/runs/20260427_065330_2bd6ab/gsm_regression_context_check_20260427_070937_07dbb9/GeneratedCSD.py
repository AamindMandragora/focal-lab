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
        d_2_openSpanToken_: _dafny.Seq
        d_2_openSpanToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))
        d_3_openedAny_: bool
        d_3_openedAny_ = insideConstrained
        d_4_openDelay_: int
        d_4_openDelay_ = 8
        d_5_constrainedTokens_: int
        d_5_constrainedTokens_ = len(currentConstrainedOut)
        d_6_maxConstrainedTokens_: int
        d_6_maxConstrainedTokens_ = 7
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_constrainedTokens_ = 0
                        if d_3_openedAny_:
                            d_7_nextAfter_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_7_nextAfter_ = out0_
                            if (d_7_nextAfter_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_nextAfter_]))
                                d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_remainingOpen_: int
                            d_8_remainingOpen_ = (maxSteps) - (d_1_steps_)
                            if ((len(generated)) < ((len(generatedPrefix)) + (d_4_openDelay_))) or ((d_8_remainingOpen_) < (3)):
                                d_9_nextPlain_: _dafny.Seq
                                out1_: _dafny.Seq
                                out1_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                d_9_nextPlain_ = out1_
                                if (d_9_nextPlain_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_nextPlain_]))
                                    d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                (lm).GenerateLogits((prompt) + (generated))
                                (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([d_2_openSpanToken_]), _dafny.BigRational('1e2'))
                                (d_0_helpers_).ScaleAllLogits(lm, _dafny.BigRational('1e0'))
                                d_10_topOpen_: _dafny.Seq
                                out2_: _dafny.Seq
                                out2_ = (d_0_helpers_).GetHighestLogitToken(lm)
                                d_10_topOpen_ = out2_
                                if VerifiedDecoderAgent.default__.Contains(d_10_topOpen_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_11_gOpen_: _dafny.Seq
                                    d_12_iOpen_: bool
                                    d_13_cOpen_: _dafny.Seq
                                    out3_: _dafny.Seq
                                    out4_: bool
                                    out5_: _dafny.Seq
                                    out3_, out4_, out5_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_11_gOpen_ = out3_
                                    d_12_iOpen_ = out4_
                                    d_13_cOpen_ = out5_
                                    generated = d_11_gOpen_
                                    insideConstrainedOut = d_12_iOpen_
                                    currentConstrainedOut = d_13_cOpen_
                                    d_3_openedAny_ = True
                                    d_5_constrainedTokens_ = len(currentConstrainedOut)
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_14_nextPlain2_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out6_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                    d_14_nextPlain2_ = out6_
                                    if (d_14_nextPlain2_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_14_nextPlain2_]))
                                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_15_completeNow_: bool
                        d_15_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_15_completeNow_:
                            d_16_gClose_: _dafny.Seq
                            d_17_iClose_: bool
                            d_18_cClose_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_16_gClose_ = out7_
                            d_17_iClose_ = out8_
                            d_18_cClose_ = out9_
                            generated = d_16_gClose_
                            insideConstrainedOut = d_17_iClose_
                            currentConstrainedOut = d_18_cClose_
                            d_5_constrainedTokens_ = 0
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            if (d_5_constrainedTokens_) >= (d_6_maxConstrainedTokens_):
                                raise _dafny.Break("0")
                            elif True:
                                d_19_nextConstrained_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                d_19_nextConstrained_ = out10_
                                if (d_19_nextConstrained_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_20_validNext_: bool
                                    out11_: bool
                                    out11_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_19_nextConstrained_)
                                    d_20_validNext_ = out11_
                                    if d_20_validNext_:
                                        d_21_gApp_: _dafny.Seq
                                        d_22_iApp_: bool
                                        d_23_cApp_: _dafny.Seq
                                        out12_: _dafny.Seq
                                        out13_: bool
                                        out14_: _dafny.Seq
                                        out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_nextConstrained_)
                                        d_21_gApp_ = out12_
                                        d_22_iApp_ = out13_
                                        d_23_cApp_ = out14_
                                        generated = d_21_gApp_
                                        insideConstrainedOut = d_22_iApp_
                                        currentConstrainedOut = d_23_cApp_
                                        d_5_constrainedTokens_ = (d_5_constrainedTokens_) + (1)
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

