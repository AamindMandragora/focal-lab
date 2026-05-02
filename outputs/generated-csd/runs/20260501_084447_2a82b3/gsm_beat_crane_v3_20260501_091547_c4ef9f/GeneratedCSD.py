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
        d_3_completedSpan_: bool
        d_3_completedSpan_ = False
        d_4_openTokSeq_: _dafny.Seq
        d_4_openTokSeq_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))])
        d_5_eosSeq_: _dafny.Seq
        d_5_eosSeq_ = _dafny.SeqWithoutIsStrInference([eosToken])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_wantOpen_: bool
                        d_6_wantOpen_ = (((not(d_2_sawSpan_)) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens))) and (((d_1_steps_) + (1)) < (maxSteps))) and ((len(generated)) <= ((len(generatedPrefix)) + (8)))
                        if d_6_wantOpen_:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).BoostTokenLogits(lm, d_4_openTokSeq_, _dafny.BigRational('1e2'))
                            d_7_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (lm).ChooseNextTokenUnconstrained()
                            d_7_next_ = out0_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_8_openedGenerated_: _dafny.Seq
                                    d_9_openedInside_: bool
                                    d_10_openedCurrent_: _dafny.Seq
                                    out1_: _dafny.Seq
                                    out2_: bool
                                    out3_: _dafny.Seq
                                    out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_8_openedGenerated_ = out1_
                                    d_9_openedInside_ = out2_
                                    d_10_openedCurrent_ = out3_
                                    generated = d_8_openedGenerated_
                                    insideConstrainedOut = d_9_openedInside_
                                    currentConstrainedOut = d_10_openedCurrent_
                                    d_2_sawSpan_ = True
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                        elif True:
                            (lm).GenerateLogits((prompt) + (generated))
                            if d_3_completedSpan_:
                                if (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens):
                                    (d_0_helpers_).PenalizeTokenLogits(lm, d_4_openTokSeq_, _dafny.BigRational('8e0'))
                                (d_0_helpers_).BoostTokenLogits(lm, d_5_eosSeq_, _dafny.BigRational('3e0'))
                            d_11_next2_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (lm).ChooseNextTokenUnconstrained()
                            d_11_next2_ = out4_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_11_next2_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                if (((d_11_next2_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and (not(d_2_sawSpan_))) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)):
                                    d_12_openedGenerated2_: _dafny.Seq
                                    d_13_openedInside2_: bool
                                    d_14_openedCurrent2_: _dafny.Seq
                                    out5_: _dafny.Seq
                                    out6_: bool
                                    out7_: _dafny.Seq
                                    out5_, out6_, out7_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_12_openedGenerated2_ = out5_
                                    d_13_openedInside2_ = out6_
                                    d_14_openedCurrent2_ = out7_
                                    generated = d_12_openedGenerated2_
                                    insideConstrainedOut = d_13_openedInside2_
                                    currentConstrainedOut = d_14_openedCurrent2_
                                    d_2_sawSpan_ = True
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next2_]))
                    elif True:
                        d_15_isComplete_: bool
                        d_15_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_15_isComplete_:
                            d_16_closedGenerated_: _dafny.Seq
                            d_17_closedInside_: bool
                            d_18_closedCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_16_closedGenerated_ = out8_
                            d_17_closedInside_ = out9_
                            d_18_closedCurrent_ = out10_
                            generated = d_16_closedGenerated_
                            insideConstrainedOut = d_17_closedInside_
                            currentConstrainedOut = d_18_closedCurrent_
                            d_3_completedSpan_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_19_stablePrefix_: _dafny.Seq
                            d_19_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_20_next3_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            if (len(validTokenGroups)) > (0):
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, (prompt) + (d_19_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                                d_20_next3_ = out11_
                            elif True:
                                (lm).GenerateLogits(((prompt) + (d_19_stablePrefix_)) + (currentConstrainedOut))
                                out12_: _dafny.Seq
                                out12_ = (d_0_helpers_).ConstrainedSample(lm, parser, currentConstrainedOut, eosToken)
                                d_20_next3_ = out12_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_20_next3_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_21_appendedGenerated_: _dafny.Seq
                                d_22_appendedInside_: bool
                                d_23_appendedCurrent_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next3_)
                                d_21_appendedGenerated_ = out13_
                                d_22_appendedInside_ = out14_
                                d_23_appendedCurrent_ = out15_
                                generated = d_21_appendedGenerated_
                                insideConstrainedOut = d_22_appendedInside_
                                currentConstrainedOut = d_23_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

