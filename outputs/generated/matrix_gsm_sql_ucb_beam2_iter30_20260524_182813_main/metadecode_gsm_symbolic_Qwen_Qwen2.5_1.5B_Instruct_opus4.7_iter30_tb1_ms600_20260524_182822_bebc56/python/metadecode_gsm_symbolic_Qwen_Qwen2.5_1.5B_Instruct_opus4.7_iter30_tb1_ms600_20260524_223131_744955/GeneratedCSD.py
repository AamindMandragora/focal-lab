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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Show every arithmetic calculation as <<expression=result>>. Examples: <<3+4=7>>, <<24/4=6>>, <<x*k=x*k>>, <<m*x/(m+n)=(m*x)//(m+n)>>. Inside << >> use ONLY digits, plain variable names (write n not {n}, write x not {x}), the operators + - * / // and parentheses, and exactly one '=' before the result. Never write { or } inside << >>. Always close every << with >> before continuing. End with: The answer is <<expression=result>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_penaltyToks_: _dafny.Seq
        d_2_penaltyToks_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "{")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "}")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " {")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " }"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_3_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                            if ((d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or (((len(d_3_next_)) >= (2)) and ((_dafny.SeqWithoutIsStrInference((d_3_next_)[(len(d_3_next_)) - (2)::])) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_4_closedGenerated_: _dafny.Seq
                        d_5_closedInside_: bool
                        d_6_closedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_4_closedGenerated_ = out1_
                        d_5_closedInside_ = out2_
                        d_6_closedCurrent_ = out3_
                        generated = d_4_closedGenerated_
                        insideConstrainedOut = d_5_closedInside_
                        currentConstrainedOut = d_6_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_7_constrainedPrompt_: _dafny.Seq
                        d_7_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_8_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if (len(currentConstrainedOut)) >= (20):
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('1e1'), eosToken)
                            d_8_next_ = out4_
                        elif (len(currentConstrainedOut)) >= (8):
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))])]), _dafny.BigRational('5e0'), d_2_penaltyToks_, _dafny.BigRational('5e0'), 12, eosToken)
                            d_8_next_ = out5_
                        elif (len(currentConstrainedOut)) >= (3):
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))])]), _dafny.BigRational('4e0'), d_2_penaltyToks_, _dafny.BigRational('5e0'), 12, eosToken)
                            d_8_next_ = out6_
                        elif True:
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_penaltyToks_, _dafny.BigRational('5e0'), 12, eosToken)
                            d_8_next_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_8_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_9_appendedGenerated_: _dafny.Seq
                            d_10_appendedInside_: bool
                            d_11_appendedCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_8_next_)
                            d_9_appendedGenerated_ = out8_
                            d_10_appendedInside_ = out9_
                            d_11_appendedCurrent_ = out10_
                            generated = d_9_appendedGenerated_
                            insideConstrainedOut = d_10_appendedInside_
                            currentConstrainedOut = d_11_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

