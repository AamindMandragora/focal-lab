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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one SQL query in the required constrained span format. Prefer starting the answer span promptly and do not add explanation or multiple alternatives.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_2_flatGroups_ = out0_
        d_3_outsideWait_: int
        d_3_outsideWait_ = 0
        d_4_forceOpenAfter_: int
        d_4_forceOpenAfter_ = 12
        d_5_rollbackLimit_: int
        d_5_rollbackLimit_ = 64
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_3_outsideWait_) >= (d_4_forceOpenAfter_):
                            d_6_openedGenerated_: _dafny.Seq
                            d_7_openedInside_: bool
                            d_8_openedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_openedGenerated_ = out1_
                            d_7_openedInside_ = out2_
                            d_8_openedCurrent_ = out3_
                            generated = d_6_openedGenerated_
                            insideConstrainedOut = d_7_openedInside_
                            currentConstrainedOut = d_8_openedCurrent_
                            d_3_outsideWait_ = 0
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_next_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_9_next_ = out4_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                if (d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_3_outsideWait_ = 0
                                elif True:
                                    d_3_outsideWait_ = (d_3_outsideWait_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_10_closedGenerated_: _dafny.Seq
                        d_11_closedInside_: bool
                        d_12_closedCurrent_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_10_closedGenerated_ = out5_
                        d_11_closedInside_ = out6_
                        d_12_closedCurrent_ = out7_
                        generated = d_10_closedGenerated_
                        insideConstrainedOut = d_11_closedInside_
                        currentConstrainedOut = d_12_closedCurrent_
                        d_3_outsideWait_ = 0
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_5_rollbackLimit_):
                        d_13_rolledGenerated_: _dafny.Seq
                        d_14_rolledCurrent_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: _dafny.Seq
                        out8_, out9_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_13_rolledGenerated_ = out8_
                        d_14_rolledCurrent_ = out9_
                        generated = d_13_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_14_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_15_stablePrefix_: _dafny.Seq
                        d_15_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_16_constrainedPrompt_: _dafny.Seq
                        d_16_constrainedPrompt_ = (prompt) + (d_15_stablePrefix_)
                        d_17_next_: _dafny.Seq
                        d_17_next_ = eosToken
                        if (len(currentConstrainedOut)) < (2):
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_17_next_ = out10_
                        elif True:
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_flatGroups_, _dafny.BigRational('2e0'), 12, eosToken)
                            d_17_next_ = out11_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_17_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_18_appendedGenerated_: _dafny.Seq
                            d_19_appendedInside_: bool
                            d_20_appendedCurrent_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                            d_18_appendedGenerated_ = out12_
                            d_19_appendedInside_ = out13_
                            d_20_appendedCurrent_ = out14_
                            generated = d_18_appendedGenerated_
                            insideConstrainedOut = d_19_appendedInside_
                            currentConstrainedOut = d_20_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

