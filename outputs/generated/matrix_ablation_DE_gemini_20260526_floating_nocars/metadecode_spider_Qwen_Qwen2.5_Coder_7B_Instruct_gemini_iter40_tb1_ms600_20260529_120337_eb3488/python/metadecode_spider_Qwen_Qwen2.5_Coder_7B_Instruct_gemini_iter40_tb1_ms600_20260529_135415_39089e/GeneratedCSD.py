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
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_seenFrom_: bool
        d_2_seenFrom_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_3_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                            if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_4_enteredGenerated_: _dafny.Seq
                                d_5_enteredInside_: bool
                                d_6_enteredCurrent_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_4_enteredGenerated_ = out1_
                                d_5_enteredInside_ = out2_
                                d_6_enteredCurrent_ = out3_
                                generated = d_4_enteredGenerated_
                                insideConstrainedOut = d_5_enteredInside_
                                currentConstrainedOut = d_6_enteredCurrent_
                                d_2_seenFrom_ = False
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_7_closedGenerated_: _dafny.Seq
                        d_8_closedInside_: bool
                        d_9_closedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_7_closedGenerated_ = out4_
                        d_8_closedInside_ = out5_
                        d_9_closedCurrent_ = out6_
                        generated = d_7_closedGenerated_
                        insideConstrainedOut = d_8_closedInside_
                        currentConstrainedOut = d_9_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_10_constrainedPrompt_: _dafny.Seq
                        d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_11_next_: _dafny.Seq
                        d_11_next_ = eosToken
                        if not(d_2_seenFrom_):
                            d_12_sqlSelectKeywords_: _dafny.Seq
                            d_12_sqlSelectKeywords_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DISTINCT"))])])
                            d_13_groupsToBoost_: _dafny.Seq
                            d_13_groupsToBoost_ = (d_12_sqlSelectKeywords_) + (validTokenGroups)
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, d_13_groupsToBoost_, _dafny.BigRational('4e0'), eosToken)
                            d_11_next_ = out7_
                        elif True:
                            d_14_penaltyTokens_: _dafny.Seq
                            d_14_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER"))])
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_14_penaltyTokens_, _dafny.BigRational('5e0'), 12, eosToken)
                            d_11_next_ = out8_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_11_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_15_appendedGenerated_: _dafny.Seq
                            d_16_appendedInside_: bool
                            d_17_appendedCurrent_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                            d_15_appendedGenerated_ = out9_
                            d_16_appendedInside_ = out10_
                            d_17_appendedCurrent_ = out11_
                            generated = d_15_appendedGenerated_
                            insideConstrainedOut = d_16_appendedInside_
                            currentConstrainedOut = d_17_appendedCurrent_
                            if (not(d_2_seenFrom_)) and ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")))):
                                d_2_seenFrom_ = True
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

