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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the word problem step by step. Wrap useful intermediate symbolic expressions and the final answer in visible << >> delimiters. Inside each delimiter, write only a compact arithmetic expression or number, with no words."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        if (maxSteps) == (0):
            cost = 0
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        d_2_mathGroups_: _dafny.Seq
        d_2_mathGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "//")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "."))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "int")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ceil")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "floor"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "x")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "y")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "d")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "r")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "k")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "m")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))])])
        d_3_combinedGroups_: _dafny.Seq
        d_3_combinedGroups_ = (d_2_mathGroups_) + (validTokenGroups)
        d_4_penaltyTokens_: _dafny.Seq
        d_4_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "?")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!")), eosToken])
        d_5_steps_: int
        d_5_steps_ = 0
        while (d_5_steps_) < (maxSteps):
            if not(insideConstrainedOut):
                d_6_openedGenerated_: _dafny.Seq
                d_7_openedInside_: bool
                d_8_openedCurrent_: _dafny.Seq
                out0_: _dafny.Seq
                out1_: bool
                out2_: _dafny.Seq
                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_6_openedGenerated_ = out0_
                d_7_openedInside_ = out1_
                d_8_openedCurrent_ = out2_
                generated = d_6_openedGenerated_
                insideConstrainedOut = d_7_openedInside_
                currentConstrainedOut = d_8_openedCurrent_
                d_5_steps_ = (d_5_steps_) + (1)
            elif (parser).IsCompletePrefix(currentConstrainedOut):
                d_9_closedGenerated_: _dafny.Seq
                d_10_closedInside_: bool
                d_11_closedCurrent_: _dafny.Seq
                out3_: _dafny.Seq
                out4_: bool
                out5_: _dafny.Seq
                out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_9_closedGenerated_ = out3_
                d_10_closedInside_ = out4_
                d_11_closedCurrent_ = out5_
                generated = d_9_closedGenerated_
                insideConstrainedOut = d_10_closedInside_
                currentConstrainedOut = d_11_closedCurrent_
                d_5_steps_ = (d_5_steps_) + (1)
                cost = d_5_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            elif True:
                d_12_constrainedPrompt_: _dafny.Seq
                d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_13_nextConstrained_: _dafny.Seq
                out6_: _dafny.Seq
                out6_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, d_3_combinedGroups_, _dafny.BigRational('5e0'), d_4_penaltyTokens_, _dafny.BigRational('5e0'), 16, eosToken)
                d_13_nextConstrained_ = out6_
                d_5_steps_ = (d_5_steps_) + (1)
                if (d_13_nextConstrained_) == (eosToken):
                    cost = d_5_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                elif (d_13_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                    pass
                elif (d_13_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                    pass
                elif True:
                    d_14_validNext_: bool
                    out7_: bool
                    out7_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_13_nextConstrained_)
                    d_14_validNext_ = out7_
                    if d_14_validNext_:
                        d_15_appendedGenerated_: _dafny.Seq
                        d_16_appendedInside_: bool
                        d_17_appendedCurrent_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_nextConstrained_)
                        d_15_appendedGenerated_ = out8_
                        d_16_appendedInside_ = out9_
                        d_17_appendedCurrent_ = out10_
                        generated = d_15_appendedGenerated_
                        insideConstrainedOut = d_16_appendedInside_
                        currentConstrainedOut = d_17_appendedCurrent_
        cost = d_5_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

