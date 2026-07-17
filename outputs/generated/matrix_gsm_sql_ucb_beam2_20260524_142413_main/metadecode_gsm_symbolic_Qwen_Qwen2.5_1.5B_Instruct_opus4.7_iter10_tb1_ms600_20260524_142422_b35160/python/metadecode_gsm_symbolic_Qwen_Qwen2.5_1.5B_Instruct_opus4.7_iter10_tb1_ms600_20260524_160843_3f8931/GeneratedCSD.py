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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Compute step by step. Every arithmetic calculation must appear inline as <<expression=result>> with an '=' sign and a value inside the angle brackets. Use bare variable names (no curly braces) or concrete digits. Always close every '<<' with a matching '>>'. End with '#### answer' giving the final expression.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_effectiveCap_: int
        if (maxSteps) > (120):
            d_2_effectiveCap_ = 120
        elif True:
            d_2_effectiveCap_ = maxSteps
        d_3_prevTok_: _dafny.Seq
        d_3_prevTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        d_4_repeatCount_: int
        d_4_repeatCount_ = 0
        d_5_spanLen_: int
        d_5_spanLen_ = 0
        d_6_penaltyTokens_: _dafny.Seq
        d_6_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "{")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "}")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "_")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "@")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "#")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "$")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "%")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "&")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "~")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "^"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (d_2_effectiveCap_):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_7_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_7_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_7_next_) == (eosToken):
                            raise _dafny.Break("0")
                        if ((len(d_7_next_)) >= (2)) and ((_dafny.SeqWithoutIsStrInference((d_7_next_)[:2:])) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_5_spanLen_ = 0
                            d_3_prevTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            d_4_repeatCount_ = 0
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                            if (d_7_next_) == (d_3_prevTok_):
                                d_4_repeatCount_ = (d_4_repeatCount_) + (1)
                                if (d_4_repeatCount_) >= (4):
                                    raise _dafny.Break("0")
                            elif True:
                                d_3_prevTok_ = d_7_next_
                                d_4_repeatCount_ = 1
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_8_closedG_: _dafny.Seq
                        d_9_closedI_: bool
                        d_10_closedC_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_8_closedG_ = out1_
                        d_9_closedI_ = out2_
                        d_10_closedC_ = out3_
                        generated = d_8_closedG_
                        insideConstrainedOut = d_9_closedI_
                        currentConstrainedOut = d_10_closedC_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_5_spanLen_ = 0
                        d_3_prevTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        d_4_repeatCount_ = 0
                    elif ((d_5_spanLen_) >= (8)) or ((d_4_repeatCount_) >= (2)):
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                        insideConstrainedOut = False
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_5_spanLen_ = 0
                        d_3_prevTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        d_4_repeatCount_ = 0
                    elif True:
                        d_11_cp_: _dafny.Seq
                        d_11_cp_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_12_eqCount_: int
                        out4_: int
                        out4_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                        d_12_eqCount_ = out4_
                        d_13_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if ((d_12_eqCount_) >= (1)) and ((d_5_spanLen_) >= (3)):
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_11_cp_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))])]), _dafny.BigRational('2e1'), d_6_penaltyTokens_, _dafny.BigRational('1e1'), 16, eosToken)
                            d_13_next_ = out5_
                        elif (d_12_eqCount_) >= (1):
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_11_cp_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))])]), _dafny.BigRational('1e1'), d_6_penaltyTokens_, _dafny.BigRational('1e1'), 16, eosToken)
                            d_13_next_ = out6_
                        elif (d_5_spanLen_) >= (4):
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_11_cp_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))])]), _dafny.BigRational('14e0'), d_6_penaltyTokens_, _dafny.BigRational('1e1'), 16, eosToken)
                            d_13_next_ = out7_
                        elif True:
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_11_cp_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_6_penaltyTokens_, _dafny.BigRational('1e1'), 16, eosToken)
                            d_13_next_ = out8_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_13_next_) == (eosToken):
                            raise _dafny.Break("0")
                        if (d_13_next_) == (d_3_prevTok_):
                            d_4_repeatCount_ = (d_4_repeatCount_) + (1)
                        elif True:
                            d_3_prevTok_ = d_13_next_
                            d_4_repeatCount_ = 1
                        d_14_appG_: _dafny.Seq
                        d_15_appI_: bool
                        d_16_appC_: _dafny.Seq
                        out9_: _dafny.Seq
                        out10_: bool
                        out11_: _dafny.Seq
                        out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                        d_14_appG_ = out9_
                        d_15_appI_ = out10_
                        d_16_appC_ = out11_
                        generated = d_14_appG_
                        insideConstrainedOut = d_15_appI_
                        currentConstrainedOut = d_16_appC_
                        d_5_spanLen_ = (d_5_spanLen_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

