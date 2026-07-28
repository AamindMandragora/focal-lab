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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Wrap every intermediate calculation and the final numeric answer inside << and >> delimiters, e.g., <<3+4=7>>. Use only integer arithmetic: + - * and // (floor division). Do NOT use functions like min, max, ceil, floor, round, or abs. Do NOT use float division /. Keep each <<...>> expression short and close it with >> promptly. Put the final answer in one final <<...>> at the end.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_lastTok_: _dafny.Seq
        d_2_lastTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        d_3_repeatCount_: int
        d_3_repeatCount_ = 0
        d_4_lastInsideTok_: _dafny.Seq
        d_4_lastInsideTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        d_5_insideRepeatCount_: int
        d_5_insideRepeatCount_ = 0
        d_6_maxSpanLen_: int
        d_6_maxSpanLen_ = 40
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_7_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_7_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_7_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                            if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_lastTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                                d_3_repeatCount_ = 0
                                d_4_lastInsideTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                                d_5_insideRepeatCount_ = 0
                            elif True:
                                if (d_7_next_) == (d_2_lastTok_):
                                    d_3_repeatCount_ = (d_3_repeatCount_) + (1)
                                    if (d_3_repeatCount_) >= (5):
                                        raise _dafny.Break("0")
                                elif True:
                                    d_2_lastTok_ = d_7_next_
                                    d_3_repeatCount_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_8_closedGenerated_: _dafny.Seq
                        d_9_closedInside_: bool
                        d_10_closedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_8_closedGenerated_ = out1_
                        d_9_closedInside_ = out2_
                        d_10_closedCurrent_ = out3_
                        generated = d_8_closedGenerated_
                        insideConstrainedOut = d_9_closedInside_
                        currentConstrainedOut = d_10_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_lastTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        d_3_repeatCount_ = 0
                        d_4_lastInsideTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        d_5_insideRepeatCount_ = 0
                    elif (len(currentConstrainedOut)) >= (d_6_maxSpanLen_):
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_11_constrainedPrompt_: _dafny.Seq
                        d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_12_next_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "min")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "max")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ceil")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "floor")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "round")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "abs")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!"))]), _dafny.BigRational('6e0'), 12, eosToken)
                        d_12_next_ = out4_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_12_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            if (d_12_next_) == (d_4_lastInsideTok_):
                                d_5_insideRepeatCount_ = (d_5_insideRepeatCount_) + (1)
                                if (d_5_insideRepeatCount_) >= (4):
                                    raise _dafny.Break("0")
                            elif True:
                                d_4_lastInsideTok_ = d_12_next_
                                d_5_insideRepeatCount_ = 0
                            d_13_appendedGenerated_: _dafny.Seq
                            d_14_appendedInside_: bool
                            d_15_appendedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                            d_13_appendedGenerated_ = out5_
                            d_14_appendedInside_ = out6_
                            d_15_appendedCurrent_ = out7_
                            generated = d_13_appendedGenerated_
                            insideConstrainedOut = d_14_appendedInside_
                            currentConstrainedOut = d_15_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

