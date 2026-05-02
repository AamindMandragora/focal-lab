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
                                d_3_gOpen_: _dafny.Seq
                                d_4_inOpen_: bool
                                d_5_cOpen_: _dafny.Seq
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: _dafny.Seq
                                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_3_gOpen_ = out0_
                                d_4_inOpen_ = out1_
                                d_5_cOpen_ = out2_
                                generated = d_3_gOpen_
                                insideConstrainedOut = d_4_inOpen_
                                currentConstrainedOut = d_5_cOpen_
                                d_2_openedHere_ = True
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                raise _dafny.Break("0")
                    elif True:
                        d_6_completeNow_: bool
                        d_6_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        d_7_remaining_: int
                        d_7_remaining_ = (maxSteps) - (d_1_steps_)
                        if (d_7_remaining_) == (0):
                            raise _dafny.Break("0")
                        elif True:
                            d_8_dead_: bool
                            out3_: bool
                            out3_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                            d_8_dead_ = out3_
                            if d_8_dead_:
                                d_9_stableDead_: _dafny.Seq
                                d_9_stableDead_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_10_gRoll_: _dafny.Seq
                                d_11_cRoll_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: _dafny.Seq
                                out4_, out5_ = (d_0_helpers_).RollbackConstrainedSpan(parser, d_9_stableDead_, generated, currentConstrainedOut)
                                d_10_gRoll_ = out4_
                                d_11_cRoll_ = out5_
                                generated = d_10_gRoll_
                                currentConstrainedOut = d_11_cRoll_
                                insideConstrainedOut = True
                                d_12_completeAfterRoll_: bool
                                d_12_completeAfterRoll_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                d_13_remainingAfterRoll_: int
                                d_13_remainingAfterRoll_ = (maxSteps) - (d_1_steps_)
                                if (d_12_completeAfterRoll_) and ((d_13_remainingAfterRoll_) <= (1)):
                                    if ((d_1_steps_) + (1)) <= (maxSteps):
                                        d_14_gClose1_: _dafny.Seq
                                        d_15_inClose1_: bool
                                        d_16_cClose1_: _dafny.Seq
                                        out6_: _dafny.Seq
                                        out7_: bool
                                        out8_: _dafny.Seq
                                        out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_14_gClose1_ = out6_
                                        d_15_inClose1_ = out7_
                                        d_16_cClose1_ = out8_
                                        generated = d_14_gClose1_
                                        insideConstrainedOut = d_15_inClose1_
                                        currentConstrainedOut = d_16_cClose1_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        raise _dafny.Break("0")
                                    elif True:
                                        raise _dafny.Break("0")
                                elif True:
                                    if (stepTokenBudget) == (0):
                                        if (d_12_completeAfterRoll_) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                            d_17_gClose2_: _dafny.Seq
                                            d_18_inClose2_: bool
                                            d_19_cClose2_: _dafny.Seq
                                            out9_: _dafny.Seq
                                            out10_: bool
                                            out11_: _dafny.Seq
                                            out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_17_gClose2_ = out9_
                                            d_18_inClose2_ = out10_
                                            d_19_cClose2_ = out11_
                                            generated = d_17_gClose2_
                                            insideConstrainedOut = d_18_inClose2_
                                            currentConstrainedOut = d_19_cClose2_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                            raise _dafny.Break("0")
                                        elif True:
                                            raise _dafny.Break("0")
                                    elif True:
                                        d_20_localBudget1_: int = int(0)
                                        if d_12_completeAfterRoll_:
                                            d_20_localBudget1_ = 1
                                        elif True:
                                            d_20_localBudget1_ = stepTokenBudget
                                        d_21_stable1_: _dafny.Seq
                                        d_21_stable1_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                        d_22_constrainedPrompt1_: _dafny.Seq
                                        d_22_constrainedPrompt1_ = (prompt) + (d_21_stable1_)
                                        d_23_current1_: _dafny.Seq
                                        d_24_hitEos1_: bool
                                        d_25_used1_: int
                                        out12_: _dafny.Seq
                                        out13_: bool
                                        out14_: int
                                        out12_, out13_, out14_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_22_constrainedPrompt1_, currentConstrainedOut, d_20_localBudget1_, eosToken)
                                        d_23_current1_ = out12_
                                        d_24_hitEos1_ = out13_
                                        d_25_used1_ = out14_
                                        if (d_24_hitEos1_) or ((d_25_used1_) == (0)):
                                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                                d_26_gClose3_: _dafny.Seq
                                                d_27_inClose3_: bool
                                                d_28_cClose3_: _dafny.Seq
                                                out15_: _dafny.Seq
                                                out16_: bool
                                                out17_: _dafny.Seq
                                                out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                                d_26_gClose3_ = out15_
                                                d_27_inClose3_ = out16_
                                                d_28_cClose3_ = out17_
                                                generated = d_26_gClose3_
                                                insideConstrainedOut = d_27_inClose3_
                                                currentConstrainedOut = d_28_cClose3_
                                                d_1_steps_ = (d_1_steps_) + (1)
                                                raise _dafny.Break("0")
                                            elif True:
                                                raise _dafny.Break("0")
                                        elif True:
                                            if ((d_1_steps_) + (d_25_used1_)) <= (maxSteps):
                                                generated = (d_21_stable1_) + (d_23_current1_)
                                                currentConstrainedOut = d_23_current1_
                                                insideConstrainedOut = True
                                                d_1_steps_ = (d_1_steps_) + (d_25_used1_)
                                            elif True:
                                                raise _dafny.Break("0")
                            elif True:
                                if (d_6_completeNow_) and ((d_7_remaining_) <= (1)):
                                    if ((d_1_steps_) + (1)) <= (maxSteps):
                                        d_29_gClose4_: _dafny.Seq
                                        d_30_inClose4_: bool
                                        d_31_cClose4_: _dafny.Seq
                                        out18_: _dafny.Seq
                                        out19_: bool
                                        out20_: _dafny.Seq
                                        out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                        d_29_gClose4_ = out18_
                                        d_30_inClose4_ = out19_
                                        d_31_cClose4_ = out20_
                                        generated = d_29_gClose4_
                                        insideConstrainedOut = d_30_inClose4_
                                        currentConstrainedOut = d_31_cClose4_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        raise _dafny.Break("0")
                                    elif True:
                                        raise _dafny.Break("0")
                                elif True:
                                    if (stepTokenBudget) == (0):
                                        if (d_6_completeNow_) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                            d_32_gClose5_: _dafny.Seq
                                            d_33_inClose5_: bool
                                            d_34_cClose5_: _dafny.Seq
                                            out21_: _dafny.Seq
                                            out22_: bool
                                            out23_: _dafny.Seq
                                            out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                            d_32_gClose5_ = out21_
                                            d_33_inClose5_ = out22_
                                            d_34_cClose5_ = out23_
                                            generated = d_32_gClose5_
                                            insideConstrainedOut = d_33_inClose5_
                                            currentConstrainedOut = d_34_cClose5_
                                            d_1_steps_ = (d_1_steps_) + (1)
                                            raise _dafny.Break("0")
                                        elif True:
                                            raise _dafny.Break("0")
                                    elif True:
                                        d_35_localBudget2_: int = int(0)
                                        if d_6_completeNow_:
                                            d_35_localBudget2_ = 1
                                        elif True:
                                            d_35_localBudget2_ = stepTokenBudget
                                        d_36_stable2_: _dafny.Seq
                                        d_36_stable2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                        d_37_constrainedPrompt2_: _dafny.Seq
                                        d_37_constrainedPrompt2_ = (prompt) + (d_36_stable2_)
                                        d_38_current2_: _dafny.Seq
                                        d_39_hitEos2_: bool
                                        d_40_used2_: int
                                        out24_: _dafny.Seq
                                        out25_: bool
                                        out26_: int
                                        out24_, out25_, out26_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_37_constrainedPrompt2_, currentConstrainedOut, d_35_localBudget2_, eosToken)
                                        d_38_current2_ = out24_
                                        d_39_hitEos2_ = out25_
                                        d_40_used2_ = out26_
                                        if (d_39_hitEos2_) or ((d_40_used2_) == (0)):
                                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                                d_41_gClose6_: _dafny.Seq
                                                d_42_inClose6_: bool
                                                d_43_cClose6_: _dafny.Seq
                                                out27_: _dafny.Seq
                                                out28_: bool
                                                out29_: _dafny.Seq
                                                out27_, out28_, out29_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                                d_41_gClose6_ = out27_
                                                d_42_inClose6_ = out28_
                                                d_43_cClose6_ = out29_
                                                generated = d_41_gClose6_
                                                insideConstrainedOut = d_42_inClose6_
                                                currentConstrainedOut = d_43_cClose6_
                                                d_1_steps_ = (d_1_steps_) + (1)
                                                raise _dafny.Break("0")
                                            elif True:
                                                raise _dafny.Break("0")
                                        elif True:
                                            if ((d_1_steps_) + (d_40_used2_)) <= (maxSteps):
                                                generated = (d_36_stable2_) + (d_38_current2_)
                                                currentConstrainedOut = d_38_current2_
                                                insideConstrainedOut = True
                                                d_1_steps_ = (d_1_steps_) + (d_40_used2_)
                                            elif True:
                                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

