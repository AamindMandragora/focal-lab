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
        d_1_mathGroups_: _dafny.Seq
        d_1_mathGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ".")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ","))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "[")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "]"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "####"))])])
        d_2_penaltyTokens_: _dafny.Seq
        d_2_penaltyTokens_ = (generatedPrefix) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
        if (maxSteps) != (0):
            d_3_steps_: int
            d_3_steps_ = 0
            d_4_done_: bool
            d_4_done_ = False
            if not(insideConstrainedOut):
                d_5_openedGenerated_: _dafny.Seq
                d_6_openedInside_: bool
                d_7_openedCurrent_: _dafny.Seq
                out0_: _dafny.Seq
                out1_: bool
                out2_: _dafny.Seq
                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_5_openedGenerated_ = out0_
                d_6_openedInside_ = out1_
                d_7_openedCurrent_ = out2_
                generated = d_5_openedGenerated_
                insideConstrainedOut = d_6_openedInside_
                currentConstrainedOut = d_7_openedCurrent_
                d_3_steps_ = (d_3_steps_) + (1)
            elif (parser).IsCompletePrefix(currentConstrainedOut):
                d_8_initialClosedGenerated_: _dafny.Seq
                d_9_initialClosedInside_: bool
                d_10_initialClosedCurrent_: _dafny.Seq
                out3_: _dafny.Seq
                out4_: bool
                out5_: _dafny.Seq
                out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_8_initialClosedGenerated_ = out3_
                d_9_initialClosedInside_ = out4_
                d_10_initialClosedCurrent_ = out5_
                generated = d_8_initialClosedGenerated_
                insideConstrainedOut = d_9_initialClosedInside_
                currentConstrainedOut = d_10_initialClosedCurrent_
                d_3_steps_ = (d_3_steps_) + (1)
                d_4_done_ = True
            elif True:
                d_11_initialPrompt_: _dafny.Seq
                d_11_initialPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_12_initialGroups_: _dafny.Seq
                d_12_initialGroups_ = (validTokenGroups) + (d_1_mathGroups_)
                d_13_initialNext_: _dafny.Seq
                out6_: _dafny.Seq
                out6_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_11_initialPrompt_, currentConstrainedOut, d_12_initialGroups_, _dafny.BigRational('4e0'), d_2_penaltyTokens_, _dafny.BigRational('8e0'), 20, eosToken)
                d_13_initialNext_ = out6_
                d_3_steps_ = (d_3_steps_) + (1)
                if (d_13_initialNext_) == (eosToken):
                    d_4_done_ = True
                elif True:
                    d_14_initialValid_: bool
                    out7_: bool
                    out7_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_13_initialNext_)
                    d_14_initialValid_ = out7_
                    if d_14_initialValid_:
                        d_15_initialAppendedGenerated_: _dafny.Seq
                        d_16_initialAppendedInside_: bool
                        d_17_initialAppendedCurrent_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_initialNext_)
                        d_15_initialAppendedGenerated_ = out8_
                        d_16_initialAppendedInside_ = out9_
                        d_17_initialAppendedCurrent_ = out10_
                        generated = d_15_initialAppendedGenerated_
                        insideConstrainedOut = d_16_initialAppendedInside_
                        currentConstrainedOut = d_17_initialAppendedCurrent_
            while ((d_3_steps_) < (maxSteps)) and (not(d_4_done_)):
                if not(insideConstrainedOut):
                    d_18_nextFree_: _dafny.Seq
                    out11_: _dafny.Seq
                    out11_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_18_nextFree_ = out11_
                    d_3_steps_ = (d_3_steps_) + (1)
                    if (d_18_nextFree_) == (eosToken):
                        d_4_done_ = True
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_18_nextFree_]))
                        if (d_18_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            d_19_enteredGenerated_: _dafny.Seq
                            d_20_enteredInside_: bool
                            d_21_enteredCurrent_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_19_enteredGenerated_ = out12_
                            d_20_enteredInside_ = out13_
                            d_21_enteredCurrent_ = out14_
                            generated = d_19_enteredGenerated_
                            insideConstrainedOut = d_20_enteredInside_
                            currentConstrainedOut = d_21_enteredCurrent_
                elif (parser).IsCompletePrefix(currentConstrainedOut):
                    d_22_closedGenerated_: _dafny.Seq
                    d_23_closedInside_: bool
                    d_24_closedCurrent_: _dafny.Seq
                    out15_: _dafny.Seq
                    out16_: bool
                    out17_: _dafny.Seq
                    out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_22_closedGenerated_ = out15_
                    d_23_closedInside_ = out16_
                    d_24_closedCurrent_ = out17_
                    generated = d_22_closedGenerated_
                    insideConstrainedOut = d_23_closedInside_
                    currentConstrainedOut = d_24_closedCurrent_
                    d_3_steps_ = (d_3_steps_) + (1)
                    d_4_done_ = True
                elif True:
                    d_25_constrainedPrompt_: _dafny.Seq
                    d_25_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_26_combinedGroups_: _dafny.Seq
                    d_26_combinedGroups_ = (validTokenGroups) + (d_1_mathGroups_)
                    d_27_nextConstrained_: _dafny.Seq
                    out18_: _dafny.Seq
                    out18_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_25_constrainedPrompt_, currentConstrainedOut, d_26_combinedGroups_, _dafny.BigRational('4e0'), d_2_penaltyTokens_, _dafny.BigRational('8e0'), 20, eosToken)
                    d_27_nextConstrained_ = out18_
                    d_3_steps_ = (d_3_steps_) + (1)
                    if (d_27_nextConstrained_) == (eosToken):
                        d_4_done_ = True
                    elif True:
                        d_28_validNext_: bool
                        out19_: bool
                        out19_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_27_nextConstrained_)
                        d_28_validNext_ = out19_
                        if d_28_validNext_:
                            d_29_appendedGenerated_: _dafny.Seq
                            d_30_appendedInside_: bool
                            d_31_appendedCurrent_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: bool
                            out22_: _dafny.Seq
                            out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_27_nextConstrained_)
                            d_29_appendedGenerated_ = out20_
                            d_30_appendedInside_ = out21_
                            d_31_appendedCurrent_ = out22_
                            generated = d_29_appendedGenerated_
                            insideConstrainedOut = d_30_appendedInside_
                            currentConstrainedOut = d_31_appendedCurrent_
            cost = d_3_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

