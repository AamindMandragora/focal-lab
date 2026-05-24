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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one SQL query in a single visible constrained span and no explanation.")))
        (d_0_helpers_).SetNonDeterministic(lm, False)
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openCount_: int = int(0)
        out0_: int
        out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generatedPrefix, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        d_2_openCount_ = out0_
        d_3_openedAnySpan_: bool
        d_3_openedAnySpan_ = insideConstrained
        if not(d_3_openedAnySpan_):
            d_3_openedAnySpan_ = (d_2_openCount_) > (0)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_3_openedAnySpan_):
                            d_4_openedGenerated_: _dafny.Seq
                            d_5_openedInside_: bool
                            d_6_openedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_openedGenerated_ = out1_
                            d_5_openedInside_ = out2_
                            d_6_openedCurrent_ = out3_
                            generated = d_4_openedGenerated_
                            insideConstrainedOut = d_5_openedInside_
                            currentConstrainedOut = d_6_openedCurrent_
                            d_3_openedAnySpan_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_nextFree_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_7_nextFree_ = out4_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_nextFree_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_nextFree_]))
                                if (d_7_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                elif True:
                                    raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_8_closedGenerated_: _dafny.Seq
                        d_9_closedInside_: bool
                        d_10_closedCurrent_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_8_closedGenerated_ = out5_
                        d_9_closedInside_ = out6_
                        d_10_closedCurrent_ = out7_
                        generated = d_8_closedGenerated_
                        insideConstrainedOut = d_9_closedInside_
                        currentConstrainedOut = d_10_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_11_stablePrefix_: _dafny.Seq
                        d_11_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_12_constrainedPrompt_: _dafny.Seq
                        d_12_constrainedPrompt_ = (prompt) + (d_11_stablePrefix_)
                        d_13_candidates_: _dafny.Seq
                        out8_: _dafny.Seq
                        out8_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, 4, eosToken)
                        d_13_candidates_ = out8_
                        d_14_next_: _dafny.Seq
                        d_14_next_ = (d_13_candidates_)[0]
                        if ((d_14_next_) == (eosToken)) and ((len(d_13_candidates_)) > (1)):
                            d_14_next_ = (d_13_candidates_)[1]
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_14_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_15_appendedGenerated_: _dafny.Seq
                            d_16_appendedInside_: bool
                            d_17_appendedCurrent_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                            d_15_appendedGenerated_ = out9_
                            d_16_appendedInside_ = out10_
                            d_17_appendedCurrent_ = out11_
                            generated = d_15_appendedGenerated_
                            insideConstrainedOut = d_16_appendedInside_
                            currentConstrainedOut = d_17_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

