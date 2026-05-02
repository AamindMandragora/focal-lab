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
        if True:
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
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('1e2'))
                            (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('8e0'))
                            d_2_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (lm).ChooseNextToken()
                            d_2_next_ = out0_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_2_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif (d_2_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                        elif True:
                            d_3_completeNow_: bool
                            d_3_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_3_completeNow_:
                                d_4_g1_: _dafny.Seq
                                d_5_i1_: bool
                                d_6_c1_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_4_g1_ = out1_
                                d_5_i1_ = out2_
                                d_6_c1_ = out3_
                                generated = d_4_g1_
                                insideConstrainedOut = d_5_i1_
                                currentConstrainedOut = d_6_c1_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                (lm).GenerateLogits((prompt) + (generated))
                                if (0) < (len(currentConstrainedOut)):
                                    d_7_lastTok_: _dafny.Seq
                                    d_7_lastTok_ = (currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]
                                    if (d_7_lastTok_) in ((lm).Tokens):
                                        (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([d_7_lastTok_]), _dafny.BigRational('3e0'))
                                d_8_candidates_: _dafny.Seq
                                out4_: _dafny.Seq
                                out4_ = (d_0_helpers_).TopValidCandidates(lm, parser, (prompt) + (generated), currentConstrainedOut, 6, eosToken)
                                d_8_candidates_ = out4_
                                d_9_chosen_: _dafny.Seq
                                d_9_chosen_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                                if (0) < (len(d_8_candidates_)):
                                    d_10_cand0_: _dafny.Seq
                                    d_10_cand0_ = (d_8_candidates_)[0]
                                    d_11_ok0_: bool
                                    out5_: bool
                                    out5_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_10_cand0_)
                                    d_11_ok0_ = out5_
                                    if d_11_ok0_:
                                        d_12_p0_: _dafny.Seq
                                        d_12_p0_ = (currentConstrainedOut) + (_dafny.SeqWithoutIsStrInference([d_10_cand0_]))
                                        d_13_done0_: bool
                                        d_13_done0_ = (parser).IsCompletePrefix(d_12_p0_)
                                        if d_13_done0_:
                                            d_9_chosen_ = d_10_cand0_
                                    if ((d_9_chosen_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "")))) and ((1) < (len(d_8_candidates_))):
                                        d_14_cand1_: _dafny.Seq
                                        d_14_cand1_ = (d_8_candidates_)[1]
                                        d_15_ok1_: bool
                                        out6_: bool
                                        out6_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_14_cand1_)
                                        d_15_ok1_ = out6_
                                        if d_15_ok1_:
                                            d_16_p1_: _dafny.Seq
                                            d_16_p1_ = (currentConstrainedOut) + (_dafny.SeqWithoutIsStrInference([d_14_cand1_]))
                                            d_17_done1_: bool
                                            d_17_done1_ = (parser).IsCompletePrefix(d_16_p1_)
                                            if d_17_done1_:
                                                d_9_chosen_ = d_14_cand1_
                                    if ((d_9_chosen_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "")))) and ((2) < (len(d_8_candidates_))):
                                        d_18_cand2_: _dafny.Seq
                                        d_18_cand2_ = (d_8_candidates_)[2]
                                        d_19_ok2_: bool
                                        out7_: bool
                                        out7_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_18_cand2_)
                                        d_19_ok2_ = out7_
                                        if d_19_ok2_:
                                            d_20_p2_: _dafny.Seq
                                            d_20_p2_ = (currentConstrainedOut) + (_dafny.SeqWithoutIsStrInference([d_18_cand2_]))
                                            d_21_done2_: bool
                                            d_21_done2_ = (parser).IsCompletePrefix(d_20_p2_)
                                            if d_21_done2_:
                                                d_9_chosen_ = d_18_cand2_
                                d_22_next_: _dafny.Seq
                                d_22_next_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                                if (d_9_chosen_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))):
                                    d_22_next_ = d_9_chosen_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    out8_: _dafny.Seq
                                    out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, (prompt) + (generated), currentConstrainedOut, eosToken)
                                    d_22_next_ = out8_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                if (d_22_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_23_g2_: _dafny.Seq
                                    d_24_i2_: bool
                                    d_25_c2_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                    d_23_g2_ = out9_
                                    d_24_i2_ = out10_
                                    d_25_c2_ = out11_
                                    generated = d_23_g2_
                                    insideConstrainedOut = d_24_i2_
                                    currentConstrainedOut = d_25_c2_
                        pass
                pass
            cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

