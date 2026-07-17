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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a valid SMILES string for the requested molecular class. Keep the constrained molecular content parser-valid at every step.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_openCount_: int
        d_2_openCount_ = 0
        d_3_closeCount_: int
        d_3_closeCount_ = 0
        d_4_openedAnySpan_: bool
        d_4_openedAnySpan_ = insideConstrained
        if not(d_4_openedAnySpan_):
            out0_: int
            out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generatedPrefix, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
            d_2_openCount_ = out0_
            out1_: int
            out1_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generatedPrefix, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
            d_3_closeCount_ = out1_
            if (d_2_openCount_) > (d_3_closeCount_):
                d_4_openedAnySpan_ = True
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_4_openedAnySpan_):
                            d_5_openedGenerated_: _dafny.Seq
                            d_6_openedInside_: bool
                            d_7_openedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_openedGenerated_ = out2_
                            d_6_openedInside_ = out3_
                            d_7_openedCurrent_ = out4_
                            generated = d_5_openedGenerated_
                            insideConstrainedOut = d_6_openedInside_
                            currentConstrainedOut = d_7_openedCurrent_
                            d_4_openedAnySpan_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_next_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next_ = out5_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_9_enteredGenerated_: _dafny.Seq
                                    d_10_enteredInside_: bool
                                    d_11_enteredCurrent_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out7_: bool
                                    out8_: _dafny.Seq
                                    out6_, out7_, out8_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_9_enteredGenerated_ = out6_
                                    d_10_enteredInside_ = out7_
                                    d_11_enteredCurrent_ = out8_
                                    generated = d_9_enteredGenerated_
                                    insideConstrainedOut = d_10_enteredInside_
                                    currentConstrainedOut = d_11_enteredCurrent_
                                    d_4_openedAnySpan_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_12_closedGenerated_: _dafny.Seq
                        d_13_closedInside_: bool
                        d_14_closedCurrent_: _dafny.Seq
                        out9_: _dafny.Seq
                        out10_: bool
                        out11_: _dafny.Seq
                        out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_12_closedGenerated_ = out9_
                        d_13_closedInside_ = out10_
                        d_14_closedCurrent_ = out11_
                        generated = d_12_closedGenerated_
                        insideConstrainedOut = d_13_closedInside_
                        currentConstrainedOut = d_14_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_15_stablePrefix_: _dafny.Seq
                        d_15_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_16_constrainedPrompt_: _dafny.Seq
                        d_16_constrainedPrompt_ = (prompt) + (d_15_stablePrefix_)
                        d_17_remaining_: int
                        d_17_remaining_ = (maxSteps) - (d_1_steps_)
                        d_18_validCount_: int
                        out12_: int
                        out12_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_18_validCount_ = out12_
                        d_19_lastTok_: _dafny.Seq
                        d_19_lastTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        d_20_repeatCount_: int
                        d_20_repeatCount_ = 0
                        if (len(currentConstrainedOut)) > (0):
                            d_19_lastTok_ = (currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]
                            out13_: int
                            out13_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, d_19_lastTok_)
                            d_20_repeatCount_ = out13_
                        if (((stepTokenBudget) > (1)) and ((d_17_remaining_) > (1))) and ((d_18_validCount_) > (8)):
                            d_21_symbolBudget_: int
                            d_21_symbolBudget_ = stepTokenBudget
                            if (d_17_remaining_) < (d_21_symbolBudget_):
                                d_21_symbolBudget_ = d_17_remaining_
                            d_22_symbolGenerated_: _dafny.Seq
                            d_23_symbolOut_: _dafny.Seq
                            d_24_hitEos_: bool
                            d_25_stepsUsed_: int
                            out14_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: int
                            out14_, out15_, out16_, out17_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_16_constrainedPrompt_, generated, currentConstrainedOut, d_21_symbolBudget_, eosToken)
                            d_22_symbolGenerated_ = out14_
                            d_23_symbolOut_ = out15_
                            d_24_hitEos_ = out16_
                            d_25_stepsUsed_ = out17_
                            generated = d_22_symbolGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_23_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_25_stepsUsed_)
                            if d_24_hitEos_:
                                raise _dafny.Break("0")
                        elif True:
                            d_26_nextIn_: _dafny.Seq
                            d_26_nextIn_ = eosToken
                            if (d_20_repeatCount_) >= (2):
                                out18_: _dafny.Seq
                                out18_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                d_26_nextIn_ = out18_
                            elif (d_18_validCount_) <= (12):
                                out19_: _dafny.Seq
                                out19_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_26_nextIn_ = out19_
                            elif True:
                                out20_: _dafny.Seq
                                out20_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), eosToken)
                                d_26_nextIn_ = out20_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_26_nextIn_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_27_appendedGenerated_: _dafny.Seq
                                d_28_appendedInside_: bool
                                d_29_appendedCurrent_: _dafny.Seq
                                out21_: _dafny.Seq
                                out22_: bool
                                out23_: _dafny.Seq
                                out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_nextIn_)
                                d_27_appendedGenerated_ = out21_
                                d_28_appendedInside_ = out22_
                                d_29_appendedCurrent_ = out23_
                                generated = d_27_appendedGenerated_
                                insideConstrainedOut = d_28_appendedInside_
                                currentConstrainedOut = d_29_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

