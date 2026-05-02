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
        d_4_freeTokens_: int
        d_4_freeTokens_ = len(generatedPrefix)
        d_5_recentMathCue_: bool
        d_5_recentMathCue_ = False
        d_6_warmup_: int
        d_6_warmup_ = 4
        d_7_cueTokens_: _dafny.Seq
        d_7_cueTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "compute")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "calculat")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "calculate")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sum")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "then")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "so")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore"))])
        d_8_openTokens_: _dafny.Seq
        d_8_openTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))])
        d_9_closePenaltyTokens_: _dafny.Seq
        d_9_closePenaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_3_completedSpan_:
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).PenalizeTokenLogits(lm, d_9_closePenaltyTokens_, _dafny.BigRational('8e0'))
                            (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('3e0'))
                            d_10_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (lm).ChooseNextTokenUnconstrained()
                            d_10_next_ = out0_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_10_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_next_]))
                                d_4_freeTokens_ = (d_4_freeTokens_) + (1)
                                d_5_recentMathCue_ = (d_10_next_) in (d_7_cueTokens_)
                        elif True:
                            d_11_shouldOpen_: bool
                            d_11_shouldOpen_ = ((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)) and ((d_4_freeTokens_) >= (d_6_warmup_))) and (((d_1_steps_) + (2)) < (maxSteps))) and ((d_5_recentMathCue_) or ((d_4_freeTokens_) >= ((d_6_warmup_) + (3))))
                            if d_11_shouldOpen_:
                                (lm).GenerateLogits((prompt) + (generated))
                                (d_0_helpers_).BoostTokenLogits(lm, d_8_openTokens_, _dafny.BigRational('4e0'))
                                d_12_topTok_: _dafny.Seq
                                out1_: _dafny.Seq
                                out1_ = (d_0_helpers_).GetHighestLogitToken(lm)
                                d_12_topTok_ = out1_
                                if (d_12_topTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_13_openedGenerated_: _dafny.Seq
                                    d_14_openedInside_: bool
                                    d_15_openedCurrent_: _dafny.Seq
                                    out2_: _dafny.Seq
                                    out3_: bool
                                    out4_: _dafny.Seq
                                    out2_, out3_, out4_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_13_openedGenerated_ = out2_
                                    d_14_openedInside_ = out3_
                                    d_15_openedCurrent_ = out4_
                                    generated = d_13_openedGenerated_
                                    insideConstrainedOut = d_14_openedInside_
                                    currentConstrainedOut = d_15_openedCurrent_
                                    d_2_sawSpan_ = True
                                    d_5_recentMathCue_ = False
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    d_16_next_: _dafny.Seq
                                    out5_: _dafny.Seq
                                    out5_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                    d_16_next_ = out5_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_16_next_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_16_next_]))
                                        d_4_freeTokens_ = (d_4_freeTokens_) + (1)
                                        if ((d_16_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and (not(d_2_sawSpan_)):
                                            insideConstrainedOut = True
                                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                            d_2_sawSpan_ = True
                                            d_5_recentMathCue_ = False
                                        elif True:
                                            d_5_recentMathCue_ = ((d_16_next_) in (d_7_cueTokens_)) or (VerifiedDecoderAgent.default__.Contains(d_16_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))))
                            elif True:
                                d_17_next_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                                d_17_next_ = out6_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_17_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_17_next_]))
                                    d_4_freeTokens_ = (d_4_freeTokens_) + (1)
                                    if ((d_17_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) and (not(d_2_sawSpan_)):
                                        insideConstrainedOut = True
                                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                        d_2_sawSpan_ = True
                                        d_5_recentMathCue_ = False
                                    elif True:
                                        d_5_recentMathCue_ = ((d_17_next_) in (d_7_cueTokens_)) or (VerifiedDecoderAgent.default__.Contains(d_17_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))))
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_18_closedGenerated_: _dafny.Seq
                            d_19_closedInside_: bool
                            d_20_closedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_18_closedGenerated_ = out7_
                            d_19_closedInside_ = out8_
                            d_20_closedCurrent_ = out9_
                            generated = d_18_closedGenerated_
                            insideConstrainedOut = d_19_closedInside_
                            currentConstrainedOut = d_20_closedCurrent_
                            d_3_completedSpan_ = True
                            d_5_recentMathCue_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_21_stablePrefix_: _dafny.Seq
                            d_21_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_22_next_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, (prompt) + (d_21_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                            d_22_next_ = out10_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_22_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_23_appendedGenerated_: _dafny.Seq
                                d_24_appendedInside_: bool
                                d_25_appendedCurrent_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                d_23_appendedGenerated_ = out11_
                                d_24_appendedInside_ = out12_
                                d_25_appendedCurrent_ = out13_
                                generated = d_23_appendedGenerated_
                                insideConstrainedOut = d_24_appendedInside_
                                currentConstrainedOut = d_25_appendedCurrent_
                                d_5_recentMathCue_ = False
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

