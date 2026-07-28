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
        (d_0_helpers_).AppendTaskGuidance(lm, ((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You must respond with exactly one line in this format:\n"))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL: <<SELECT ... FROM ...>>\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "The SQL query goes between << and >>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use only tables and columns from the schema. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "No semicolon. No markdown. No explanation."))))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_freeSteps_: int
        d_2_freeSteps_ = 0
        d_3_maxFreeSteps_: int
        d_3_maxFreeSteps_ = 6
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_2_freeSteps_) < (d_3_maxFreeSteps_):
                            d_4_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_4_next_ = out0_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_freeSteps_ = (d_2_freeSteps_) + (1)
                            if (d_4_next_) == (eosToken):
                                raise _dafny.Break("0")
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            d_5_g2_: _dafny.Seq
                            d_6_i2_: bool
                            d_7_c2_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_g2_ = out1_
                            d_6_i2_ = out2_
                            d_7_c2_ = out3_
                            generated = d_5_g2_
                            insideConstrainedOut = d_6_i2_
                            currentConstrainedOut = d_7_c2_
                            d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_8_closedGenerated_: _dafny.Seq
                        d_9_closedInside_: bool
                        d_10_closedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_8_closedGenerated_ = out4_
                        d_9_closedInside_ = out5_
                        d_10_closedCurrent_ = out6_
                        generated = d_8_closedGenerated_
                        insideConstrainedOut = d_9_closedInside_
                        currentConstrainedOut = d_10_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_11_constrainedPrompt_: _dafny.Seq
                        d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_12_next_: _dafny.Seq
                        d_13_wasConstrained_: bool
                        out7_: _dafny.Seq
                        out8_: bool
                        out7_, out8_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_12_next_ = out7_
                        d_13_wasConstrained_ = out8_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_12_next_) == (eosToken):
                            raise _dafny.Break("0")
                        d_14_appendedGenerated_: _dafny.Seq
                        d_15_appendedInside_: bool
                        d_16_appendedCurrent_: _dafny.Seq
                        out9_: _dafny.Seq
                        out10_: bool
                        out11_: _dafny.Seq
                        out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                        d_14_appendedGenerated_ = out9_
                        d_15_appendedInside_ = out10_
                        d_16_appendedCurrent_ = out11_
                        generated = d_14_appendedGenerated_
                        insideConstrainedOut = d_15_appendedInside_
                        currentConstrainedOut = d_16_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

