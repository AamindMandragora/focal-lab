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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_1_steps_) + (1)) <= (maxSteps):
                            d_2_gOpen_: _dafny.Seq
                            d_3_inOpen_: bool
                            d_4_cOpen_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_2_gOpen_ = out0_
                            d_3_inOpen_ = out1_
                            d_4_cOpen_ = out2_
                            generated = d_2_gOpen_
                            insideConstrainedOut = d_3_inOpen_
                            currentConstrainedOut = d_4_cOpen_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        d_5_complete_: bool
                        d_5_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_5_complete_:
                            if ((d_1_steps_) + (1)) <= (maxSteps):
                                d_6_gClose_: _dafny.Seq
                                d_7_inClose_: bool
                                d_8_cClose_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_6_gClose_ = out3_
                                d_7_inClose_ = out4_
                                d_8_cClose_ = out5_
                                generated = d_6_gClose_
                                insideConstrainedOut = d_7_inClose_
                                currentConstrainedOut = d_8_cClose_
                                d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_9_dead_: bool
                            out6_: bool
                            out6_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_9_dead_ = out6_
                            if d_9_dead_:
                                d_10_stableDead_: _dafny.Seq
                                d_10_stableDead_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_11_gRoll_: _dafny.Seq
                                d_12_cRoll_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: _dafny.Seq
                                out7_, out8_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_10_stableDead_, generated, currentConstrainedOut)
                                d_11_gRoll_ = out7_
                                d_12_cRoll_ = out8_
                                generated = d_11_gRoll_
                                currentConstrainedOut = d_12_cRoll_
                                insideConstrainedOut = True
                                d_13_completeRolled_: bool
                                d_13_completeRolled_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if d_13_completeRolled_:
                                    if ((d_1_steps_) + (1)) <= (maxSteps):
                                        d_14_gClose2_: _dafny.Seq
                                        d_15_inClose2_: bool
                                        d_16_cClose2_: _dafny.Seq
                                        out9_: _dafny.Seq
                                        out10_: bool
                                        out11_: _dafny.Seq
                                        out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_14_gClose2_ = out9_
                                        d_15_inClose2_ = out10_
                                        d_16_cClose2_ = out11_
                                        generated = d_14_gClose2_
                                        insideConstrainedOut = d_15_inClose2_
                                        currentConstrainedOut = d_16_cClose2_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        raise _dafny.Break("0")
                                    elif True:
                                        raise _dafny.Break("0")
                                elif True:
                                    d_17_remaining1_: int
                                    d_17_remaining1_ = (maxSteps) - (d_1_steps_)
                                    if ((d_17_remaining1_) == (0)) or ((stepTokenBudget) == (0)):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_18_stable1_: _dafny.Seq
                                        d_18_stable1_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                        d_19_constrainedPrompt1_: _dafny.Seq
                                        d_19_constrainedPrompt1_ = (prompt) + (d_18_stable1_)
                                        d_20_current1_: _dafny.Seq
                                        d_21_hitEos1_: bool
                                        d_22_used1_: int
                                        out12_: _dafny.Seq
                                        out13_: bool
                                        out14_: int
                                        out12_, out13_, out14_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_19_constrainedPrompt1_, currentConstrainedOut, stepTokenBudget, eosToken)
                                        d_20_current1_ = out12_
                                        d_21_hitEos1_ = out13_
                                        d_22_used1_ = out14_
                                        if (d_21_hitEos1_) or ((d_22_used1_) == (0)):
                                            raise _dafny.Break("0")
                                        elif True:
                                            if ((d_1_steps_) + (d_22_used1_)) <= (maxSteps):
                                                generated = (d_18_stable1_) + (d_20_current1_)
                                                currentConstrainedOut = d_20_current1_
                                                insideConstrainedOut = True
                                                d_1_steps_ = (d_1_steps_) + (d_22_used1_)
                                            elif True:
                                                raise _dafny.Break("0")
                            elif True:
                                d_23_remaining2_: int
                                d_23_remaining2_ = (maxSteps) - (d_1_steps_)
                                if ((d_23_remaining2_) == (0)) or ((stepTokenBudget) == (0)):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_24_stable2_: _dafny.Seq
                                    d_24_stable2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                    d_25_constrainedPrompt2_: _dafny.Seq
                                    d_25_constrainedPrompt2_ = (prompt) + (d_24_stable2_)
                                    d_26_current2_: _dafny.Seq
                                    d_27_hitEos2_: bool
                                    d_28_used2_: int
                                    out15_: _dafny.Seq
                                    out16_: bool
                                    out17_: int
                                    out15_, out16_, out17_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_25_constrainedPrompt2_, currentConstrainedOut, stepTokenBudget, eosToken)
                                    d_26_current2_ = out15_
                                    d_27_hitEos2_ = out16_
                                    d_28_used2_ = out17_
                                    if (d_27_hitEos2_) or ((d_28_used2_) == (0)):
                                        raise _dafny.Break("0")
                                    elif True:
                                        if ((d_1_steps_) + (d_28_used2_)) <= (maxSteps):
                                            generated = (d_24_stable2_) + (d_26_current2_)
                                            currentConstrainedOut = d_26_current2_
                                            insideConstrainedOut = True
                                            d_1_steps_ = (d_1_steps_) + (d_28_used2_)
                                        elif True:
                                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

