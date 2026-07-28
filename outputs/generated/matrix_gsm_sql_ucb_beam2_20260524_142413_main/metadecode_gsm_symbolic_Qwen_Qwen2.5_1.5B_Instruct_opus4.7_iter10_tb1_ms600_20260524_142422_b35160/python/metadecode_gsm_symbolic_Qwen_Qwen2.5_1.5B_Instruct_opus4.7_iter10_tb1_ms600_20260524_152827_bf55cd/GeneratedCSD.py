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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the problem step by step using the variable names that appear in the problem statement. FORMAT REQUIREMENT: every arithmetic calculation in your reasoning must be written inline as <<expression=result>>, where the equals sign and the resulting value both appear inside the double-angle brackets. Always pair every '<<' with a matching '>>', and always place an '=' between the expression and its result. After the calculations, write a single concluding line of the form '#### N', where N is the final numeric answer. Do not copy any phrasing from these instructions verbatim into your answer.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_effectiveCap_: int
        if (maxSteps) > (350):
            d_2_effectiveCap_ = 350
        elif True:
            d_2_effectiveCap_ = maxSteps
        with _dafny.label("0"):
            while (d_1_steps_) < (d_2_effectiveCap_):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_remaining_: int
                        d_3_remaining_ = (d_2_effectiveCap_) - (d_1_steps_)
                        d_4_chunkBudget_: int
                        if (d_3_remaining_) > (80):
                            d_4_chunkBudget_ = 80
                        elif True:
                            d_4_chunkBudget_ = d_3_remaining_
                        d_5_chunkedG_: _dafny.Seq
                        d_6_stoppedOpen_: bool
                        d_7_stoppedEos_: bool
                        d_8_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_5_chunkedG_ = out0_
                        d_6_stoppedOpen_ = out1_
                        d_7_stoppedEos_ = out2_
                        d_8_stepsUsed_ = out3_
                        generated = d_5_chunkedG_
                        d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                        if d_7_stoppedEos_:
                            raise _dafny.Break("0")
                        if d_6_stoppedOpen_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_9_closedG_: _dafny.Seq
                        d_10_closedI_: bool
                        d_11_closedC_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_9_closedG_ = out4_
                        d_10_closedI_ = out5_
                        d_11_closedC_ = out6_
                        generated = d_9_closedG_
                        insideConstrainedOut = d_10_closedI_
                        currentConstrainedOut = d_11_closedC_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_12_cp_: _dafny.Seq
                        d_12_cp_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_13_eqCount_: int
                        out7_: int
                        out7_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                        d_13_eqCount_ = out7_
                        d_14_spanLen_: int
                        d_14_spanLen_ = len(currentConstrainedOut)
                        d_15_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if ((d_13_eqCount_) == (0)) and ((d_14_spanLen_) >= (3)):
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_12_cp_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))]), _dafny.BigRational('6e0'), eosToken)
                            d_15_next_ = out8_
                        elif ((d_13_eqCount_) >= (1)) and ((d_14_spanLen_) >= (6)):
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_12_cp_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('8e0'), eosToken)
                            d_15_next_ = out9_
                        elif True:
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_12_cp_, currentConstrainedOut, eosToken)
                            d_15_next_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_15_next_) == (eosToken):
                            raise _dafny.Break("0")
                        d_16_appG_: _dafny.Seq
                        d_17_appI_: bool
                        d_18_appC_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: _dafny.Seq
                        out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                        d_16_appG_ = out11_
                        d_17_appI_ = out12_
                        d_18_appC_ = out13_
                        generated = d_16_appG_
                        insideConstrainedOut = d_17_appI_
                        currentConstrainedOut = d_18_appC_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

