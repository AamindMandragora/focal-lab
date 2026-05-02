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
                            d_9_constrainedPrompt_: _dafny.Seq
                            d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_10_narrow_: bool
                            out6_: bool
                            out6_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 4)
                            d_10_narrow_ = out6_
                            if d_10_narrow_:
                                d_11_next_: _dafny.Seq
                                out7_: _dafny.Seq
                                out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_11_next_ = out7_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_11_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_12_appendedGenerated_: _dafny.Seq
                                    d_13_appendedInside_: bool
                                    d_14_appendedCurrent_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out9_: bool
                                    out10_: _dafny.Seq
                                    out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                                    d_12_appendedGenerated_ = out8_
                                    d_13_appendedInside_ = out9_
                                    d_14_appendedCurrent_ = out10_
                                    generated = d_12_appendedGenerated_
                                    insideConstrainedOut = d_13_appendedInside_
                                    currentConstrainedOut = d_14_appendedCurrent_
                            elif True:
                                d_15_candidates_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, 8, eosToken)
                                d_15_candidates_ = out11_
                                if (len(d_15_candidates_)) == (0):
                                    d_16_next2_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out12_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_16_next2_ = out12_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_16_next2_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_17_appendedGenerated2_: _dafny.Seq
                                        d_18_appendedInside2_: bool
                                        d_19_appendedCurrent2_: _dafny.Seq
                                        out13_: _dafny.Seq
                                        out14_: bool
                                        out15_: _dafny.Seq
                                        out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next2_)
                                        d_17_appendedGenerated2_ = out13_
                                        d_18_appendedInside2_ = out14_
                                        d_19_appendedCurrent2_ = out15_
                                        generated = d_17_appendedGenerated2_
                                        insideConstrainedOut = d_18_appendedInside2_
                                        currentConstrainedOut = d_19_appendedCurrent2_
                                elif True:
                                    (lm).GenerateLogits((d_9_constrainedPrompt_) + (currentConstrainedOut))
                                    d_20_best_: _dafny.Seq
                                    d_20_best_ = (d_15_candidates_)[0]
                                    d_21_bestLogit_: _dafny.BigRational
                                    out16_: _dafny.BigRational
                                    out16_ = (d_0_helpers_).GetTokenLogit(lm, d_20_best_)
                                    d_21_bestLogit_ = out16_
                                    d_22_i_: int
                                    d_22_i_ = 1
                                    while (d_22_i_) < (len(d_15_candidates_)):
                                        d_23_tok_: _dafny.Seq
                                        d_23_tok_ = (d_15_candidates_)[d_22_i_]
                                        d_24_tokLogit_: _dafny.BigRational
                                        out17_: _dafny.BigRational
                                        out17_ = (d_0_helpers_).GetTokenLogit(lm, d_23_tok_)
                                        d_24_tokLogit_ = out17_
                                        if (d_24_tokLogit_) > (d_21_bestLogit_):
                                            d_20_best_ = d_23_tok_
                                            d_21_bestLogit_ = d_24_tokLogit_
                                        d_22_i_ = (d_22_i_) + (1)
                                    d_25_bestValid_: bool
                                    out18_: bool
                                    out18_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_20_best_)
                                    d_25_bestValid_ = out18_
                                    if d_25_bestValid_:
                                        d_26_appendedGenerated3_: _dafny.Seq
                                        d_27_appendedInside3_: bool
                                        d_28_appendedCurrent3_: _dafny.Seq
                                        out19_: _dafny.Seq
                                        out20_: bool
                                        out21_: _dafny.Seq
                                        out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_best_)
                                        d_26_appendedGenerated3_ = out19_
                                        d_27_appendedInside3_ = out20_
                                        d_28_appendedCurrent3_ = out21_
                                        generated = d_26_appendedGenerated3_
                                        insideConstrainedOut = d_27_appendedInside3_
                                        currentConstrainedOut = d_28_appendedCurrent3_
                                    d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

