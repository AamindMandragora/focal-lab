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
        d_2_sawSpan_: bool
        d_2_sawSpan_ = insideConstrained
        d_3_forceOpenAfter_: int
        d_3_forceOpenAfter_ = 6
        d_4_mustOpenBy_: int
        d_4_mustOpenBy_ = 18
        d_5_stopBiasTokens_: _dafny.Seq
        d_5_stopBiasTokens_ = _dafny.SeqWithoutIsStrInference([eosToken])
        d_6_canOpen_: bool
        d_6_canOpen_ = (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_7_closedGenerated_: _dafny.Seq
                            d_8_closedInside_: bool
                            d_9_closedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_7_closedGenerated_ = out0_
                            d_8_closedInside_ = out1_
                            d_9_closedCurrent_ = out2_
                            generated = d_7_closedGenerated_
                            insideConstrainedOut = d_8_closedInside_
                            currentConstrainedOut = d_9_closedCurrent_
                            d_2_sawSpan_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_10_stablePrefix_: _dafny.Seq
                            d_10_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_11_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_10_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 8, eosToken)
                            d_11_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_11_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_12_appendedGenerated_: _dafny.Seq
                                d_13_appendedInside_: bool
                                d_14_appendedCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                                d_12_appendedGenerated_ = out4_
                                d_13_appendedInside_ = out5_
                                d_14_appendedCurrent_ = out6_
                                generated = d_12_appendedGenerated_
                                insideConstrainedOut = d_13_appendedInside_
                                currentConstrainedOut = d_14_appendedCurrent_
                    elif True:
                        d_15_generatedSinceStart_: int
                        d_15_generatedSinceStart_ = (len(generated)) - (len(generatedPrefix))
                        if ((d_6_canOpen_) and (not(d_2_sawSpan_))) and ((d_15_generatedSinceStart_) >= (d_3_forceOpenAfter_)):
                            d_16_openedGenerated_: _dafny.Seq
                            d_17_openedInside_: bool
                            d_18_openedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_16_openedGenerated_ = out7_
                            d_17_openedInside_ = out8_
                            d_18_openedCurrent_ = out9_
                            generated = d_16_openedGenerated_
                            insideConstrainedOut = d_17_openedInside_
                            currentConstrainedOut = d_18_openedCurrent_
                            d_2_sawSpan_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            if ((d_6_canOpen_) and (not(d_2_sawSpan_))) and (((d_15_generatedSinceStart_) + (1)) >= (maxSteps)):
                                d_19_openedGenerated2_: _dafny.Seq
                                d_20_openedInside2_: bool
                                d_21_openedCurrent2_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_19_openedGenerated2_ = out10_
                                d_20_openedInside2_ = out11_
                                d_21_openedCurrent2_ = out12_
                                generated = d_19_openedGenerated2_
                                insideConstrainedOut = d_20_openedInside2_
                                currentConstrainedOut = d_21_openedCurrent2_
                                d_2_sawSpan_ = True
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                if d_2_sawSpan_:
                                    (lm).GenerateLogits((prompt) + (generated))
                                    (d_0_helpers_).BoostTokenLogits(lm, d_5_stopBiasTokens_, _dafny.BigRational('1e2'))
                                    d_22_next2_: _dafny.Seq
                                    out13_: _dafny.Seq
                                    out13_ = (lm).ChooseNextTokenUnconstrained()
                                    d_22_next2_ = out13_
                                    (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_22_next2_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        if ((d_22_next2_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and (d_6_canOpen_):
                                            d_23_openedGenerated3_: _dafny.Seq
                                            d_24_openedInside3_: bool
                                            d_25_openedCurrent3_: _dafny.Seq
                                            out14_: _dafny.Seq
                                            out15_: bool
                                            out16_: _dafny.Seq
                                            out14_, out15_, out16_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                            d_23_openedGenerated3_ = out14_
                                            d_24_openedInside3_ = out15_
                                            d_25_openedCurrent3_ = out16_
                                            generated = d_23_openedGenerated3_
                                            insideConstrainedOut = d_24_openedInside3_
                                            currentConstrainedOut = d_25_openedCurrent3_
                                            d_2_sawSpan_ = True
                                        elif True:
                                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_22_next2_]))
                                elif True:
                                    if (d_6_canOpen_) and ((d_15_generatedSinceStart_) >= (d_4_mustOpenBy_)):
                                        d_26_openedGenerated4_: _dafny.Seq
                                        d_27_openedInside4_: bool
                                        d_28_openedCurrent4_: _dafny.Seq
                                        out17_: _dafny.Seq
                                        out18_: bool
                                        out19_: _dafny.Seq
                                        out17_, out18_, out19_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                        d_26_openedGenerated4_ = out17_
                                        d_27_openedInside4_ = out18_
                                        d_28_openedCurrent4_ = out19_
                                        generated = d_26_openedGenerated4_
                                        insideConstrainedOut = d_27_openedInside4_
                                        currentConstrainedOut = d_28_openedCurrent4_
                                        d_2_sawSpan_ = True
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        d_29_next3_: _dafny.Seq
                                        out20_: _dafny.Seq
                                        out20_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                        d_29_next3_ = out20_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        if (d_29_next3_) == (eosToken):
                                            raise _dafny.Break("0")
                                        elif True:
                                            if ((d_29_next3_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and (d_6_canOpen_):
                                                d_30_openedGenerated5_: _dafny.Seq
                                                d_31_openedInside5_: bool
                                                d_32_openedCurrent5_: _dafny.Seq
                                                out21_: _dafny.Seq
                                                out22_: bool
                                                out23_: _dafny.Seq
                                                out21_, out22_, out23_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                                d_30_openedGenerated5_ = out21_
                                                d_31_openedInside5_ = out22_
                                                d_32_openedCurrent5_ = out23_
                                                generated = d_30_openedGenerated5_
                                                insideConstrainedOut = d_31_openedInside5_
                                                currentConstrainedOut = d_32_openedCurrent5_
                                                d_2_sawSpan_ = True
                                            elif True:
                                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_29_next3_]))
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

