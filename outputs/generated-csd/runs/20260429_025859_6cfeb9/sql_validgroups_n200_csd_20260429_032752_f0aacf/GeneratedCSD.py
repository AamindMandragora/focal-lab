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
                        d_2_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_2_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_2_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                            if (d_2_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_3_completeNow_: bool
                        d_3_completeNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_3_completeNow_:
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
                            d_7_validCount_: int
                            out4_: int
                            out4_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_7_validCount_ = out4_
                            if (d_7_validCount_) == (0):
                                d_8_repaired_: _dafny.Seq
                                out5_: _dafny.Seq
                                out5_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")))
                                d_8_repaired_ = out5_
                                if (len(d_8_repaired_)) == (len(currentConstrainedOut)):
                                    out6_: _dafny.Seq
                                    out6_ = VerifiedDecoderAgent.CSDHelpers.RollbackToBoundary(parser, currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")))
                                    d_8_repaired_ = out6_
                                if (len(d_8_repaired_)) == (len(currentConstrainedOut)):
                                    out7_: _dafny.Seq
                                    out7_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrainedOut)
                                    d_8_repaired_ = out7_
                                d_9_stablePrefix_: _dafny.Seq
                                d_9_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                generated = (d_9_stablePrefix_) + (d_8_repaired_)
                                currentConstrainedOut = d_8_repaired_
                            elif True:
                                (lm).GenerateLogits((prompt) + (generated))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_10_next2_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).GetHighestLogitToken(lm)
                                d_10_next2_ = out8_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_10_next2_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_11_next2Valid_: bool
                                    out9_: bool
                                    out9_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_10_next2_)
                                    d_11_next2Valid_ = out9_
                                    if d_11_next2Valid_:
                                        d_12_appendedGenerated_: _dafny.Seq
                                        d_13_appendedInside_: bool
                                        d_14_appendedCurrent_: _dafny.Seq
                                        out10_: _dafny.Seq
                                        out11_: bool
                                        out12_: _dafny.Seq
                                        out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next2_)
                                        d_12_appendedGenerated_ = out10_
                                        d_13_appendedInside_ = out11_
                                        d_14_appendedCurrent_ = out12_
                                        generated = d_12_appendedGenerated_
                                        insideConstrainedOut = d_13_appendedInside_
                                        currentConstrainedOut = d_14_appendedCurrent_
                                    elif True:
                                        raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

