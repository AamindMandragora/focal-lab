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
        d_2_recentMathCue_: bool
        d_2_recentMathCue_ = False
        d_3_sawAnySpan_: bool
        d_3_sawAnySpan_ = insideConstrained
        d_4_minOpenDelay_: int
        d_4_minOpenDelay_ = 3
        d_5_mathCueTokens_: _dafny.Seq
        d_5_mathCueTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "compute")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "calculate")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sum")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "difference")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "product")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "quotient")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "then")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "so"))])
        d_6_openSpanTokens_: _dafny.Seq
        d_6_openSpanTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_7_generatedSinceStart_: int
                        d_7_generatedSinceStart_ = (len(generated)) - (len(generatedPrefix))
                        if (((not(d_3_sawAnySpan_)) and (d_2_recentMathCue_)) and ((d_7_generatedSinceStart_) >= (d_4_minOpenDelay_))) and ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))) in ((lm).Tokens)):
                            (lm).GenerateLogits((prompt) + (generated))
                            (d_0_helpers_).BoostTokenLogits(lm, d_6_openSpanTokens_, _dafny.BigRational('8e0'))
                            d_8_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (lm).ChooseNextTokenUnconstrained()
                            d_8_next_ = out0_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_recentMathCue_ = False
                                    d_3_sawAnySpan_ = True
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                    d_2_recentMathCue_ = ((d_8_next_) in (d_5_mathCueTokens_)) or (VerifiedDecoderAgent.default__.Contains(d_8_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))))
                        elif True:
                            d_9_next_: _dafny.Seq
                            out1_: _dafny.Seq
                            out1_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_9_next_ = out1_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                if (d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_recentMathCue_ = False
                                    d_3_sawAnySpan_ = True
                                elif True:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                    d_2_recentMathCue_ = ((d_9_next_) in (d_5_mathCueTokens_)) or (VerifiedDecoderAgent.default__.Contains(d_9_next_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))))
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_10_closedGenerated_: _dafny.Seq
                            d_11_closedInside_: bool
                            d_12_closedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_10_closedGenerated_ = out2_
                            d_11_closedInside_ = out3_
                            d_12_closedCurrent_ = out4_
                            generated = d_10_closedGenerated_
                            insideConstrainedOut = d_11_closedInside_
                            currentConstrainedOut = d_12_closedCurrent_
                            d_2_recentMathCue_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_13_stablePrefix_: _dafny.Seq
                            d_13_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_14_next_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, (prompt) + (d_13_stablePrefix_), currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_14_next_ = out5_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_14_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_15_appendedGenerated_: _dafny.Seq
                                d_16_appendedInside_: bool
                                d_17_appendedCurrent_: _dafny.Seq
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: _dafny.Seq
                                out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                d_15_appendedGenerated_ = out6_
                                d_16_appendedInside_ = out7_
                                d_17_appendedCurrent_ = out8_
                                generated = d_15_appendedGenerated_
                                insideConstrainedOut = d_16_appendedInside_
                                currentConstrainedOut = d_17_appendedCurrent_
                                d_2_recentMathCue_ = False
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

