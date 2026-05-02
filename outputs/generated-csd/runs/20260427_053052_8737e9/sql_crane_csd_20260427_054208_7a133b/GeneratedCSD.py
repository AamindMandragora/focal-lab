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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, eosToken):
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
                        d_2_chunkBudget_: int
                        d_2_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_3_chunkedGenerated_: _dafny.Seq
                        d_4_stoppedOpen_: bool
                        d_5_stoppedEos_: bool
                        d_6_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_3_chunkedGenerated_ = out0_
                        d_4_stoppedOpen_ = out1_
                        d_5_stoppedEos_ = out2_
                        d_6_stepsUsed_ = out3_
                        generated = d_3_chunkedGenerated_
                        d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed_)
                        if d_5_stoppedEos_:
                            raise _dafny.Break("0")
                        elif True:
                            if d_4_stoppedOpen_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_7_isComplete_: bool
                        d_7_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_7_isComplete_:
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
                        elif True:
                            d_11_candidates_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).TopValidCandidates(lm, parser, prompt, currentConstrainedOut, 8, eosToken)
                            d_11_candidates_ = out7_
                            if (len(d_11_candidates_)) == (0):
                                raise _dafny.Break("0")
                            elif True:
                                d_12_stablePrefix_: _dafny.Seq
                                d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                (lm).GenerateLogits(((prompt) + (d_12_stablePrefix_)) + (currentConstrainedOut))
                                (d_0_helpers_).BoostTokenLogits(lm, d_11_candidates_, _dafny.BigRational('1e2'))
                                (d_0_helpers_).PenalizeTokenLogits(lm, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('1e2'))
                                d_13_next_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (lm).ChooseNextToken()
                                d_13_next_ = out8_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                if (d_13_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_14_validNext_: bool
                                    out9_: bool
                                    out9_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_13_next_)
                                    d_14_validNext_ = out9_
                                    if d_14_validNext_:
                                        d_15_appendedGenerated_: _dafny.Seq
                                        d_16_appendedInside_: bool
                                        d_17_appendedCurrent_: _dafny.Seq
                                        out10_: _dafny.Seq
                                        out11_: bool
                                        out12_: _dafny.Seq
                                        out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                                        d_15_appendedGenerated_ = out10_
                                        d_16_appendedInside_ = out11_
                                        d_17_appendedCurrent_ = out12_
                                        generated = d_15_appendedGenerated_
                                        insideConstrainedOut = d_16_appendedInside_
                                        currentConstrainedOut = d_17_appendedCurrent_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                    elif True:
                                        raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

