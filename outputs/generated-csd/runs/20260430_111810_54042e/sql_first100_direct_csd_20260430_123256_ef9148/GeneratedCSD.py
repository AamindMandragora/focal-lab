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
        d_2_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_2_flatGroups_ = out0_
        d_3_rollbackToken_: _dafny.Seq
        d_3_rollbackToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ","))
        if not((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ","))) in (d_2_flatGroups_)):
            d_3_rollbackToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))
        if not((d_3_rollbackToken_) in (d_2_flatGroups_)):
            d_3_rollbackToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE"))
        if not((d_3_rollbackToken_) in (d_2_flatGroups_)):
            d_3_rollbackToken_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND"))
        d_4_longPrefixThreshold_: int
        d_4_longPrefixThreshold_ = 80
        d_5_narrowThreshold_: int
        d_5_narrowThreshold_ = 2
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_next_: _dafny.Seq
                        out1_: _dafny.Seq
                        out1_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_6_next_ = out1_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_6_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                            if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_7_complete_: bool
                        d_7_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_7_complete_:
                            d_8_closedGenerated_: _dafny.Seq
                            d_9_closedInside_: bool
                            d_10_closedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_8_closedGenerated_ = out2_
                            d_9_closedInside_ = out3_
                            d_10_closedCurrent_ = out4_
                            generated = d_8_closedGenerated_
                            insideConstrainedOut = d_9_closedInside_
                            currentConstrainedOut = d_10_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_11_validCount_: int
                            out5_: int
                            out5_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_11_validCount_ = out5_
                            if ((len(currentConstrainedOut)) >= (d_4_longPrefixThreshold_)) or ((d_11_validCount_) <= (d_5_narrowThreshold_)):
                                d_12_repaired_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, d_3_rollbackToken_)
                                d_12_repaired_ = out6_
                                d_13_trimCount_: int
                                d_13_trimCount_ = (len(currentConstrainedOut)) - (len(d_12_repaired_))
                                generated = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (d_13_trimCount_):])
                                currentConstrainedOut = d_12_repaired_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_14_constrainedPrompt_: _dafny.Seq
                                d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_15_next_: _dafny.Seq
                                out7_: _dafny.Seq
                                out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_15_next_ = out7_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_15_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_16_appendedGenerated_: _dafny.Seq
                                    d_17_appendedInside_: bool
                                    d_18_appendedCurrent_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out9_: bool
                                    out10_: _dafny.Seq
                                    out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                    d_16_appendedGenerated_ = out8_
                                    d_17_appendedInside_ = out9_
                                    d_18_appendedCurrent_ = out10_
                                    generated = d_16_appendedGenerated_
                                    insideConstrainedOut = d_17_appendedInside_
                                    currentConstrainedOut = d_18_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

