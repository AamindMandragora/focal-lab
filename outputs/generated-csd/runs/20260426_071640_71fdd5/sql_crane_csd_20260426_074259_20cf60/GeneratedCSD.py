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
        if True:
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
                            d_1_steps_ = (d_1_steps_) + (1)
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            d_2_complete_: bool
                            d_2_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_2_complete_:
                                d_3_closedGenerated_: _dafny.Seq
                                d_4_closedInside_: bool
                                d_5_closedCurrent_: _dafny.Seq
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: _dafny.Seq
                                out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_3_closedGenerated_ = out0_
                                d_4_closedInside_ = out1_
                                d_5_closedCurrent_ = out2_
                                generated = d_3_closedGenerated_
                                insideConstrainedOut = d_4_closedInside_
                                currentConstrainedOut = d_5_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                d_6_validCount_: int
                                out3_: int
                                out3_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                                d_6_validCount_ = out3_
                                d_7_narrow_: bool
                                out4_: bool
                                out4_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 3)
                                d_7_narrow_ = out4_
                                d_8_constrainedPrompt_: _dafny.Seq
                                d_8_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                (lm).GenerateLogits((d_8_constrainedPrompt_) + (currentConstrainedOut))
                                if (d_6_validCount_) <= (8):
                                    d_9_cands_: _dafny.Seq
                                    out5_: _dafny.Seq
                                    out5_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 8, eosToken)
                                    d_9_cands_ = out5_
                                    (d_0_helpers_).BoostTokenLogits(lm, d_9_cands_, _dafny.BigRational('8e0'))
                                elif True:
                                    d_10_cands2_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out6_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 5, eosToken)
                                    d_10_cands2_ = out6_
                                    (d_0_helpers_).BoostTokenLogits(lm, d_10_cands2_, _dafny.BigRational('3e0'))
                                if d_7_narrow_:
                                    (d_0_helpers_).ScaleAllLogits(lm, _dafny.BigRational('5e-1'))
                                    d_11_cands3_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out7_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 10, eosToken)
                                    d_11_cands3_ = out7_
                                    (d_0_helpers_).BoostTokenLogits(lm, d_11_cands3_, _dafny.BigRational('1e2'))
                                d_12_next_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_12_next_ = out8_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_12_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_13_appendedGenerated_: _dafny.Seq
                                    d_14_appendedInside_: bool
                                    d_15_appendedCurrent_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                                    d_13_appendedGenerated_ = out9_
                                    d_14_appendedInside_ = out10_
                                    d_15_appendedCurrent_ = out11_
                                    generated = d_13_appendedGenerated_
                                    insideConstrainedOut = d_14_appendedInside_
                                    currentConstrainedOut = d_15_appendedCurrent_
                        pass
                pass
            cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

