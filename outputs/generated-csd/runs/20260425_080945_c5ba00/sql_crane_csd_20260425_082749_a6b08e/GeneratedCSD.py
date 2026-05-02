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
        d_2_hasOpened_: bool
        d_2_hasOpened_ = insideConstrained
        d_3_hasClosed_: bool
        d_3_hasClosed_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_3_hasClosed_:
                            if (d_1_steps_) < (maxSteps):
                                (lm).GenerateLogits((prompt) + (generated))
                                (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                                d_4_nextAfterClose_: _dafny.Seq
                                out0_: _dafny.Seq
                                out0_ = (lm).ChooseNextTokenUnconstrained()
                                d_4_nextAfterClose_ = out0_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_4_nextAfterClose_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_nextAfterClose_]))
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            d_5_next_: _dafny.Seq
                            out1_: _dafny.Seq
                            out1_ = (lm).ChooseNextTokenUnconstrained()
                            d_5_next_ = out1_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_5_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                                if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_hasOpened_ = True
                    elif True:
                        d_6_complete_: bool
                        d_6_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_6_complete_:
                            d_7_closedGenerated_: _dafny.Seq
                            d_8_closedInside_: bool
                            d_9_closedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_7_closedGenerated_ = out2_
                            d_8_closedInside_ = out3_
                            d_9_closedCurrent_ = out4_
                            generated = d_7_closedGenerated_
                            insideConstrainedOut = d_8_closedInside_
                            currentConstrainedOut = d_9_closedCurrent_
                            d_3_hasClosed_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_10_narrow_: bool
                            out5_: bool
                            out5_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_10_narrow_ = out5_
                            if d_10_narrow_:
                                d_11_rolled_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                d_11_rolled_ = out6_
                                generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - ((len(currentConstrainedOut)) - (len(d_11_rolled_))):])
                                currentConstrainedOut = d_11_rolled_
                            elif True:
                                d_12_constrainedPrompt_: _dafny.Seq
                                d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                (lm).GenerateLogits((d_12_constrainedPrompt_) + (currentConstrainedOut))
                                (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('1e2'))
                                d_13_argmax_: _dafny.Seq
                                out7_: _dafny.Seq
                                out7_ = (d_0_helpers_).GetHighestLogitToken(lm)
                                d_13_argmax_ = out7_
                                d_14_argmaxValid_: bool
                                out8_: bool
                                out8_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_13_argmax_)
                                d_14_argmaxValid_ = out8_
                                if (d_14_argmaxValid_) and ((d_13_argmax_) != (eosToken)):
                                    d_15_appendedGenerated_: _dafny.Seq
                                    d_16_appendedInside_: bool
                                    d_17_appendedCurrent_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_argmax_)
                                    d_15_appendedGenerated_ = out9_
                                    d_16_appendedInside_ = out10_
                                    d_17_appendedCurrent_ = out11_
                                    generated = d_15_appendedGenerated_
                                    insideConstrainedOut = d_16_appendedInside_
                                    currentConstrainedOut = d_17_appendedCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_18_nextConstrained_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out12_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_18_nextConstrained_ = out12_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_18_nextConstrained_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_19_appendedGenerated2_: _dafny.Seq
                                        d_20_appendedInside2_: bool
                                        d_21_appendedCurrent2_: _dafny.Seq
                                        out13_: _dafny.Seq
                                        out14_: bool
                                        out15_: _dafny.Seq
                                        out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_nextConstrained_)
                                        d_19_appendedGenerated2_ = out13_
                                        d_20_appendedInside2_ = out14_
                                        d_21_appendedCurrent2_ = out15_
                                        generated = d_19_appendedGenerated2_
                                        insideConstrainedOut = d_20_appendedInside2_
                                        currentConstrainedOut = d_21_appendedCurrent2_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_22_completeAtEnd_: bool
            d_22_completeAtEnd_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if d_22_completeAtEnd_:
                d_23_closedGenerated2_: _dafny.Seq
                d_24_closedInside2_: bool
                d_25_closedCurrent2_: _dafny.Seq
                out16_: _dafny.Seq
                out17_: bool
                out18_: _dafny.Seq
                out16_, out17_, out18_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_23_closedGenerated2_ = out16_
                d_24_closedInside2_ = out17_
                d_25_closedCurrent2_ = out18_
                generated = d_23_closedGenerated2_
                insideConstrainedOut = d_24_closedInside2_
                currentConstrainedOut = d_25_closedCurrent2_
                d_3_hasClosed_ = True
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

