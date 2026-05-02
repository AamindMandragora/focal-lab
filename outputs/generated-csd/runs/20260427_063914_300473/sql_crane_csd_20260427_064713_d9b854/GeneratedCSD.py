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
                        d_2_openedGenerated_: _dafny.Seq
                        d_3_openedInside_: bool
                        d_4_openedCurrent_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_2_openedGenerated_ = out0_
                        d_3_openedInside_ = out1_
                        d_4_openedCurrent_ = out2_
                        generated = d_2_openedGenerated_
                        insideConstrainedOut = d_3_openedInside_
                        currentConstrainedOut = d_4_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_5_complete_: bool
                        d_5_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_5_complete_:
                            d_6_closedGenerated_: _dafny.Seq
                            d_7_closedInside_: bool
                            d_8_closedCurrent_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_6_closedGenerated_ = out3_
                            d_7_closedInside_ = out4_
                            d_8_closedCurrent_ = out5_
                            generated = d_6_closedGenerated_
                            insideConstrainedOut = d_7_closedInside_
                            currentConstrainedOut = d_8_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_candidates_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 8, eosToken)
                            d_9_candidates_ = out6_
                            if (len(d_9_candidates_)) > (0):
                                (lm).GenerateLogits((prompt) + (generated))
                                (d_0_helpers_).BoostTokenLogits(lm, d_9_candidates_, _dafny.BigRational('8e0'))
                                (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('1e2'))
                                d_10_next_: _dafny.Seq
                                out7_: _dafny.Seq
                                out7_ = (lm).ChooseNextToken()
                                d_10_next_ = out7_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                if (d_10_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_11_isValid_: bool
                                    out8_: bool
                                    out8_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_10_next_)
                                    d_11_isValid_ = out8_
                                    if d_11_isValid_:
                                        d_12_appendedGenerated1_: _dafny.Seq
                                        d_13_appendedInside1_: bool
                                        d_14_appendedCurrent1_: _dafny.Seq
                                        out9_: _dafny.Seq
                                        out10_: bool
                                        out11_: _dafny.Seq
                                        out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                                        d_12_appendedGenerated1_ = out9_
                                        d_13_appendedInside1_ = out10_
                                        d_14_appendedCurrent1_ = out11_
                                        generated = d_12_appendedGenerated1_
                                        insideConstrainedOut = d_13_appendedInside1_
                                        currentConstrainedOut = d_14_appendedCurrent1_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        d_15_backup_: _dafny.Seq
                                        out12_: _dafny.Seq
                                        out12_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                        d_15_backup_ = out12_
                                        if (d_15_backup_) == (eosToken):
                                            raise _dafny.Break("0")
                                        elif True:
                                            d_16_appendedGenerated2_: _dafny.Seq
                                            d_17_appendedInside2_: bool
                                            d_18_appendedCurrent2_: _dafny.Seq
                                            out13_: _dafny.Seq
                                            out14_: bool
                                            out15_: _dafny.Seq
                                            out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_backup_)
                                            d_16_appendedGenerated2_ = out13_
                                            d_17_appendedInside2_ = out14_
                                            d_18_appendedCurrent2_ = out15_
                                            generated = d_16_appendedGenerated2_
                                            insideConstrainedOut = d_17_appendedInside2_
                                            currentConstrainedOut = d_18_appendedCurrent2_
                                            d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_19_next2_: _dafny.Seq
                                out16_: _dafny.Seq
                                out16_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                                d_19_next2_ = out16_
                                if (d_19_next2_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_20_appendedGenerated3_: _dafny.Seq
                                    d_21_appendedInside3_: bool
                                    d_22_appendedCurrent3_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next2_)
                                    d_20_appendedGenerated3_ = out17_
                                    d_21_appendedInside3_ = out18_
                                    d_22_appendedCurrent3_ = out19_
                                    generated = d_20_appendedGenerated3_
                                    insideConstrainedOut = d_21_appendedInside3_
                                    currentConstrainedOut = d_22_appendedCurrent3_
                                    d_1_steps_ = (d_1_steps_) + (1)
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

