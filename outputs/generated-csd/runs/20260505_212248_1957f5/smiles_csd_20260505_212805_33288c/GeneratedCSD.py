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
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        d_3_cueArmed_: bool
        d_3_cueArmed_ = False
        d_4_lastTok_: _dafny.Seq
        d_5_foundLast_: bool
        out0_: _dafny.Seq
        out1_: bool
        out0_, out1_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
        d_4_lastTok_ = out0_
        d_5_foundLast_ = out1_
        if d_5_foundLast_:
            if ((((((d_4_lastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))) or ((d_4_lastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SMILES"))))) or ((d_4_lastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "smiles"))))) or ((d_4_lastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))))) or ((d_4_lastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_4_lastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))):
                d_3_cueArmed_ = True
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_3_cueArmed_:
                            d_6_openedGenerated_: _dafny.Seq
                            d_7_openedInside_: bool
                            d_8_openedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_openedGenerated_ = out2_
                            d_7_openedInside_ = out3_
                            d_8_openedCurrent_ = out4_
                            generated = d_6_openedGenerated_
                            insideConstrainedOut = d_7_openedInside_
                            currentConstrainedOut = d_8_openedCurrent_
                            d_3_cueArmed_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_chunkBudget_: int
                            if ((maxSteps) - (d_1_steps_)) > (4):
                                d_9_chunkBudget_ = 4
                            elif True:
                                d_9_chunkBudget_ = (maxSteps) - (d_1_steps_)
                            d_10_chunkedGenerated_: _dafny.Seq
                            d_11_stoppedOnOpenSpan_: bool
                            d_12_stoppedOnEos_: bool
                            d_13_stepsUsed_: int
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: bool
                            out8_: int
                            out5_, out6_, out7_, out8_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_9_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_10_chunkedGenerated_ = out5_
                            d_11_stoppedOnOpenSpan_ = out6_
                            d_12_stoppedOnEos_ = out7_
                            d_13_stepsUsed_ = out8_
                            generated = d_10_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_13_stepsUsed_)
                            d_14_newLastTok_: _dafny.Seq
                            d_15_foundNewLast_: bool
                            out9_: _dafny.Seq
                            out10_: bool
                            out9_, out10_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                            d_14_newLastTok_ = out9_
                            d_15_foundNewLast_ = out10_
                            if d_15_foundNewLast_:
                                if ((((((d_14_newLastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))) or ((d_14_newLastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SMILES"))))) or ((d_14_newLastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "smiles"))))) or ((d_14_newLastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))))) or ((d_14_newLastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_14_newLastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))):
                                    d_3_cueArmed_ = True
                            if d_12_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_11_stoppedOnOpenSpan_:
                                d_16_enteredGenerated_: _dafny.Seq
                                d_17_enteredInside_: bool
                                d_18_enteredCurrent_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_16_enteredGenerated_ = out11_
                                d_17_enteredInside_ = out12_
                                d_18_enteredCurrent_ = out13_
                                generated = d_16_enteredGenerated_
                                insideConstrainedOut = d_17_enteredInside_
                                currentConstrainedOut = d_18_enteredCurrent_
                                d_3_cueArmed_ = False
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_19_closedGenerated_: _dafny.Seq
                        d_20_closedInside_: bool
                        d_21_closedCurrent_: _dafny.Seq
                        out14_: _dafny.Seq
                        out15_: bool
                        out16_: _dafny.Seq
                        out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_19_closedGenerated_ = out14_
                        d_20_closedInside_ = out15_
                        d_21_closedCurrent_ = out16_
                        generated = d_19_closedGenerated_
                        insideConstrainedOut = d_20_closedInside_
                        currentConstrainedOut = d_21_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_22_stablePrefix_: _dafny.Seq
                        d_22_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_23_constrainedPrompt_: _dafny.Seq
                        d_23_constrainedPrompt_ = (prompt) + (d_22_stablePrefix_)
                        d_24_validCount_: int
                        out17_: int
                        out17_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_24_validCount_ = out17_
                        if (d_24_validCount_) <= (d_2_narrowThreshold_):
                            d_25_next_: _dafny.Seq
                            out18_: _dafny.Seq
                            out18_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                            d_25_next_ = out18_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_25_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_26_appendedGenerated_: _dafny.Seq
                                d_27_appendedInside_: bool
                                d_28_appendedCurrent_: _dafny.Seq
                                out19_: _dafny.Seq
                                out20_: bool
                                out21_: _dafny.Seq
                                out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_25_next_)
                                d_26_appendedGenerated_ = out19_
                                d_27_appendedInside_ = out20_
                                d_28_appendedCurrent_ = out21_
                                generated = d_26_appendedGenerated_
                                insideConstrainedOut = d_27_appendedInside_
                                currentConstrainedOut = d_28_appendedCurrent_
                        elif True:
                            d_29_remaining_: int
                            d_29_remaining_ = (maxSteps) - (d_1_steps_)
                            d_30_symbolBudget_: int
                            if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_29_remaining_)):
                                d_30_symbolBudget_ = d_29_remaining_
                            elif True:
                                d_30_symbolBudget_ = stepTokenBudget
                            d_31_symbolGenerated_: _dafny.Seq
                            d_32_symbolCurrent_: _dafny.Seq
                            d_33_hitEos_: bool
                            d_34_usedSteps_: int
                            out22_: _dafny.Seq
                            out23_: _dafny.Seq
                            out24_: bool
                            out25_: int
                            out22_, out23_, out24_, out25_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_23_constrainedPrompt_, generated, currentConstrainedOut, d_30_symbolBudget_, eosToken)
                            d_31_symbolGenerated_ = out22_
                            d_32_symbolCurrent_ = out23_
                            d_33_hitEos_ = out24_
                            d_34_usedSteps_ = out25_
                            generated = d_31_symbolGenerated_
                            currentConstrainedOut = d_32_symbolCurrent_
                            d_1_steps_ = (d_1_steps_) + (d_34_usedSteps_)
                            if d_33_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

