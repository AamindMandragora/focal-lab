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
        d_2_chemistryArmed_: bool
        d_2_chemistryArmed_ = False
        d_3_recentAfterAnswer_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))
        d_3_recentAfterAnswer_ = out0_
        d_4_freeTokensSinceOpen_: int
        d_4_freeTokensSinceOpen_ = 0
        d_5_forceOpenAfter_: int
        d_5_forceOpenAfter_ = 6
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_2_chemistryArmed_) or ((d_4_freeTokensSinceOpen_) >= (d_5_forceOpenAfter_)):
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
                            d_2_chemistryArmed_ = False
                            d_4_freeTokensSinceOpen_ = 0
                            out4_: _dafny.Seq
                            out4_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))
                            d_3_recentAfterAnswer_ = out4_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_remainingOutside_: int
                            d_9_remainingOutside_ = (maxSteps) - (d_1_steps_)
                            d_10_chunkBudget_: int
                            if (d_9_remainingOutside_) > (3):
                                d_10_chunkBudget_ = 3
                            elif True:
                                d_10_chunkBudget_ = d_9_remainingOutside_
                            d_11_chunkedGenerated_: _dafny.Seq
                            d_12_stoppedOnOpenSpan_: bool
                            d_13_stoppedOnEos_: bool
                            d_14_stepsUsed_: int
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: bool
                            out8_: int
                            out5_, out6_, out7_, out8_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_10_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_11_chunkedGenerated_ = out5_
                            d_12_stoppedOnOpenSpan_ = out6_
                            d_13_stoppedOnEos_ = out7_
                            d_14_stepsUsed_ = out8_
                            generated = d_11_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_14_stepsUsed_)
                            d_4_freeTokensSinceOpen_ = (d_4_freeTokensSinceOpen_) + (d_14_stepsUsed_)
                            if d_13_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_12_stoppedOnOpenSpan_:
                                d_15_enteredGenerated_: _dafny.Seq
                                d_16_enteredInside_: bool
                                d_17_enteredCurrent_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_15_enteredGenerated_ = out9_
                                d_16_enteredInside_ = out10_
                                d_17_enteredCurrent_ = out11_
                                generated = d_15_enteredGenerated_
                                insideConstrainedOut = d_16_enteredInside_
                                currentConstrainedOut = d_17_enteredCurrent_
                                d_2_chemistryArmed_ = False
                                d_4_freeTokensSinceOpen_ = 0
                            elif True:
                                out12_: _dafny.Seq
                                out12_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))
                                d_3_recentAfterAnswer_ = out12_
                                if (len(d_3_recentAfterAnswer_)) > (0):
                                    d_2_chemistryArmed_ = True
                                elif True:
                                    d_18_prevTok_: _dafny.Seq
                                    d_19_foundPrev_: bool
                                    out13_: _dafny.Seq
                                    out14_: bool
                                    out13_, out14_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                                    d_18_prevTok_ = out13_
                                    d_19_foundPrev_ = out14_
                                    if d_19_foundPrev_:
                                        if ((((d_18_prevTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SMILES")))) or ((d_18_prevTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "smiles"))))) or ((d_18_prevTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "molecule"))))) or ((d_18_prevTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "molecular")))):
                                            d_2_chemistryArmed_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_20_closedGenerated_: _dafny.Seq
                        d_21_closedInside_: bool
                        d_22_closedCurrent_: _dafny.Seq
                        out15_: _dafny.Seq
                        out16_: bool
                        out17_: _dafny.Seq
                        out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_20_closedGenerated_ = out15_
                        d_21_closedInside_ = out16_
                        d_22_closedCurrent_ = out17_
                        generated = d_20_closedGenerated_
                        insideConstrainedOut = d_21_closedInside_
                        currentConstrainedOut = d_22_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_chemistryArmed_ = False
                        d_4_freeTokensSinceOpen_ = 0
                        out18_: _dafny.Seq
                        out18_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))
                        d_3_recentAfterAnswer_ = out18_
                    elif True:
                        d_23_constrainedPrompt_: _dafny.Seq
                        d_23_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_24_next_: _dafny.Seq
                        out19_: _dafny.Seq
                        out19_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_24_next_ = out19_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_24_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_25_appendedGenerated_: _dafny.Seq
                            d_26_appendedInside_: bool
                            d_27_appendedCurrent_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: bool
                            out22_: _dafny.Seq
                            out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                            d_25_appendedGenerated_ = out20_
                            d_26_appendedInside_ = out21_
                            d_27_appendedCurrent_ = out22_
                            generated = d_25_appendedGenerated_
                            insideConstrainedOut = d_26_appendedInside_
                            currentConstrainedOut = d_27_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

