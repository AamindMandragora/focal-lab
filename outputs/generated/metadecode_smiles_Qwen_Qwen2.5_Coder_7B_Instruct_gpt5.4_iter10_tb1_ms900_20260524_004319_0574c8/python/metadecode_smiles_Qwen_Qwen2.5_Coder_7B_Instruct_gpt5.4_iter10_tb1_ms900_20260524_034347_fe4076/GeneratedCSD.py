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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one valid SMILES string. You may write a short prefix before the molecule, but once the constrained molecule span begins, keep generating the molecule until complete and then close it immediately. Avoid malformed or repetitive continuations.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        d_3_repetitionStart_: int
        d_3_repetitionStart_ = 8
        d_4_outsideBudget_: int
        d_4_outsideBudget_ = 2
        d_5_outsideSteps_: int
        d_5_outsideSteps_ = 0
        d_6_openedHere_: bool
        d_6_openedHere_ = insideConstrained
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_6_openedHere_) or ((d_5_outsideSteps_) >= (d_4_outsideBudget_)):
                            d_7_openedGenerated_: _dafny.Seq
                            d_8_openedInside_: bool
                            d_9_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_7_openedGenerated_ = out0_
                            d_8_openedInside_ = out1_
                            d_9_openedCurrent_ = out2_
                            generated = d_7_openedGenerated_
                            insideConstrainedOut = d_8_openedInside_
                            currentConstrainedOut = d_9_openedCurrent_
                            d_6_openedHere_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_10_nextOutside_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_10_nextOutside_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_5_outsideSteps_ = (d_5_outsideSteps_) + (1)
                            if (d_10_nextOutside_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_nextOutside_]))
                                if (d_10_nextOutside_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_11_enteredGenerated_: _dafny.Seq
                                    d_12_enteredInside_: bool
                                    d_13_enteredCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_11_enteredGenerated_ = out4_
                                    d_12_enteredInside_ = out5_
                                    d_13_enteredCurrent_ = out6_
                                    generated = d_11_enteredGenerated_
                                    insideConstrainedOut = d_12_enteredInside_
                                    currentConstrainedOut = d_13_enteredCurrent_
                                    d_6_openedHere_ = True
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_14_closedGenerated_: _dafny.Seq
                            d_15_closedInside_: bool
                            d_16_closedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_14_closedGenerated_ = out7_
                            d_15_closedInside_ = out8_
                            d_16_closedCurrent_ = out9_
                            generated = d_14_closedGenerated_
                            insideConstrainedOut = d_15_closedInside_
                            currentConstrainedOut = d_16_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_17_constrainedPrompt_: _dafny.Seq
                            d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_18_next_: _dafny.Seq
                            d_18_next_ = eosToken
                            d_19_validCount_: int
                            out10_: int
                            out10_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_19_validCount_ = out10_
                            if (len(currentConstrainedOut)) >= (d_3_repetitionStart_):
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                d_18_next_ = out11_
                            elif (d_19_validCount_) <= (3):
                                out12_: _dafny.Seq
                                out12_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                                d_18_next_ = out12_
                            elif (d_19_validCount_) <= (d_2_narrowThreshold_):
                                out13_: _dafny.Seq
                                out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                                d_18_next_ = out13_
                            elif True:
                                d_20_nextSoft_: _dafny.Seq
                                d_21_usedFallback_: bool
                                out14_: _dafny.Seq
                                out15_: bool
                                out14_, out15_ = (d_0_helpers_).SafeSoftConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e0'), eosToken)
                                d_20_nextSoft_ = out14_
                                d_21_usedFallback_ = out15_
                                d_18_next_ = d_20_nextSoft_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_18_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_22_appendedGenerated_: _dafny.Seq
                                d_23_appendedInside_: bool
                                d_24_appendedCurrent_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                                d_22_appendedGenerated_ = out16_
                                d_23_appendedInside_ = out17_
                                d_24_appendedCurrent_ = out18_
                                generated = d_22_appendedGenerated_
                                insideConstrainedOut = d_23_appendedInside_
                                currentConstrainedOut = d_24_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

