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
                        d_5_isComplete_: bool
                        d_5_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_5_isComplete_:
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
                            raise _dafny.Break("0")
                        elif True:
                            d_9_stablePrefix_: _dafny.Seq
                            d_9_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_10_constrainedPrompt_: _dafny.Seq
                            d_10_constrainedPrompt_ = (prompt) + (d_9_stablePrefix_)
                            (lm).GenerateLogits((d_10_constrainedPrompt_) + (currentConstrainedOut))
                            if (len(validTokenGroups)) > (0):
                                d_11_flatHints_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                d_11_flatHints_ = out6_
                                if (len(d_11_flatHints_)) > (0):
                                    d_12_hintedValid_: bool
                                    out7_: bool
                                    out7_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_11_flatHints_)
                                    d_12_hintedValid_ = out7_
                            if (len(currentConstrainedOut)) > (0):
                                d_13_lastTok_: _dafny.Seq
                                d_13_lastTok_ = (currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]
                                d_14_lastStillValid_: bool
                                out8_: bool
                                out8_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_13_lastTok_)
                                d_14_lastStillValid_ = out8_
                            (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                            d_15_next_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = (lm).ChooseNextToken()
                            d_15_next_ = out9_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_15_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_16_appendedGenerated_: _dafny.Seq
                                d_17_appendedInside_: bool
                                d_18_appendedCurrent_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                d_16_appendedGenerated_ = out10_
                                d_17_appendedInside_ = out11_
                                d_18_appendedCurrent_ = out12_
                                generated = d_16_appendedGenerated_
                                insideConstrainedOut = d_17_appendedInside_
                                currentConstrainedOut = d_18_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

