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
        d_2_chemistryCueSeen_: bool
        d_2_chemistryCueSeen_ = False
        d_3_recentSmilesUpper_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SMILES")))
        d_3_recentSmilesUpper_ = out0_
        d_4_recentSmilesLower_: _dafny.Seq
        out1_: _dafny.Seq
        out1_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "smiles")))
        d_4_recentSmilesLower_ = out1_
        if ((len(d_3_recentSmilesUpper_)) > (0)) or ((len(d_4_recentSmilesLower_)) > (0)):
            d_2_chemistryCueSeen_ = True
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_2_chemistryCueSeen_:
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
                            d_2_chemistryCueSeen_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_chunkBudget_: int
                            d_8_chunkBudget_ = (maxSteps) - (d_1_steps_)
                            d_9_chunkSize_: int
                            if (d_8_chunkBudget_) <= (2):
                                d_9_chunkSize_ = d_8_chunkBudget_
                            elif True:
                                d_9_chunkSize_ = 2
                            d_10_chunkedGenerated_: _dafny.Seq
                            d_11_stoppedOpen_: bool
                            d_12_stoppedEos_: bool
                            d_13_stepsUsed_: int
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: bool
                            out8_: int
                            out5_, out6_, out7_, out8_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_9_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_10_chunkedGenerated_ = out5_
                            d_11_stoppedOpen_ = out6_
                            d_12_stoppedEos_ = out7_
                            d_13_stepsUsed_ = out8_
                            generated = d_10_chunkedGenerated_
                            d_1_steps_ = (d_1_steps_) + (d_13_stepsUsed_)
                            d_14_seenUpper_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SMILES")))
                            d_14_seenUpper_ = out9_
                            d_15_seenLower_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "smiles")))
                            d_15_seenLower_ = out10_
                            if ((len(d_14_seenUpper_)) > (0)) or ((len(d_15_seenLower_)) > (0)):
                                d_2_chemistryCueSeen_ = True
                            if d_12_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_11_stoppedOpen_:
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
                                d_2_chemistryCueSeen_ = False
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
                        d_2_chemistryCueSeen_ = False
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_22_stablePrefix_: _dafny.Seq
                        d_22_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_23_constrainedPrompt_: _dafny.Seq
                        d_23_constrainedPrompt_ = (prompt) + (d_22_stablePrefix_)
                        d_24_next_: _dafny.Seq
                        out17_: _dafny.Seq
                        out17_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_23_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_24_next_ = out17_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_24_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_25_appendedGenerated_: _dafny.Seq
                            d_26_appendedInside_: bool
                            d_27_appendedCurrent_: _dafny.Seq
                            out18_: _dafny.Seq
                            out19_: bool
                            out20_: _dafny.Seq
                            out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next_)
                            d_25_appendedGenerated_ = out18_
                            d_26_appendedInside_ = out19_
                            d_27_appendedCurrent_ = out20_
                            generated = d_25_appendedGenerated_
                            insideConstrainedOut = d_26_appendedInside_
                            currentConstrainedOut = d_27_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

