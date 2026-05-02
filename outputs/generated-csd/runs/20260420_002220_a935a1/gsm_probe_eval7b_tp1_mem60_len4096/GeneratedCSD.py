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
        while (d_1_steps_) < (maxSteps):
            if not(insideConstrainedOut):
                (lm).GenerateLogits((prompt) + (generated))
                if (len(generated)) > (0):
                    d_2_lastTok_: _dafny.Seq
                    d_2_lastTok_ = (generated)[(len(generated)) - (1)]
                    if ((((((((((VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "let"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Let"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "define"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Define"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "quantity"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Quantity"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "symbol"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Symbol"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer")))):
                        (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('8e0'))
                    if (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final")))) or (VerifiedDecoderAgent.default__.Contains(d_2_lastTok_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))):
                        (d_0_helpers_).BoostTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('3e0'))
                d_3_next_: _dafny.Seq
                out0_: _dafny.Seq
                out0_ = (lm).ChooseNextTokenUnconstrained()
                d_3_next_ = out0_
                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_3_next_) == (eosToken):
                    pass
                elif True:
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                    if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                if (d_3_next_) == (eosToken):
                    d_1_steps_ = maxSteps
            elif True:
                d_4_isComplete_: bool
                d_4_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                if d_4_isComplete_:
                    d_5_closedGenerated_: _dafny.Seq
                    d_6_closedInside_: bool
                    d_7_closedCurrent_: _dafny.Seq
                    out1_: _dafny.Seq
                    out2_: bool
                    out3_: _dafny.Seq
                    out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_5_closedGenerated_ = out1_
                    d_6_closedInside_ = out2_
                    d_7_closedCurrent_ = out3_
                    generated = d_5_closedGenerated_
                    insideConstrainedOut = d_6_closedInside_
                    currentConstrainedOut = d_7_closedCurrent_
                    d_1_steps_ = (d_1_steps_) + (1)
                elif True:
                    d_8_stablePrefix_: _dafny.Seq
                    d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                    d_9_constrainedPrompt_: _dafny.Seq
                    d_9_constrainedPrompt_ = (prompt) + (d_8_stablePrefix_)
                    d_10_next_: _dafny.Seq
                    out4_: _dafny.Seq
                    out4_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken)
                    d_10_next_ = out4_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_10_next_) == (eosToken):
                        d_1_steps_ = maxSteps
                    elif True:
                        d_11_appendedGenerated_: _dafny.Seq
                        d_12_appendedInside_: bool
                        d_13_appendedCurrent_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                        d_11_appendedGenerated_ = out5_
                        d_12_appendedInside_ = out6_
                        d_13_appendedCurrent_ = out7_
                        generated = d_11_appendedGenerated_
                        insideConstrainedOut = d_12_appendedInside_
                        currentConstrainedOut = d_13_appendedCurrent_
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

