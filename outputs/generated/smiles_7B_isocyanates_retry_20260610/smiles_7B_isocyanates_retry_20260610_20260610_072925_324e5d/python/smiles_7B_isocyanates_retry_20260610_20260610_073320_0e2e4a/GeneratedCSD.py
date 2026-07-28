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
        if (maxSteps) == (0):
            pass
        elif True:
            (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output the SMILES string for the given chemical name. Write << then the SMILES string then >>. Example: <<CCO>> for ethanol. Do not add explanation.")))
            d_1_steps_: int
            d_1_steps_ = 0
            if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_2_maxChunk_: int
                d_2_maxChunk_ = (maxSteps) - (1)
                if (d_2_maxChunk_) == (0):
                    d_2_maxChunk_ = 1
                d_3_genOut_: _dafny.Seq
                d_4_stoppedOnOpenSpan_: bool
                d_5_stoppedOnEos_: bool
                d_6_stepsUsed_: int
                out0_: _dafny.Seq
                out1_: bool
                out2_: bool
                out3_: int
                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_maxChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                d_3_genOut_ = out0_
                d_4_stoppedOnOpenSpan_ = out1_
                d_5_stoppedOnEos_ = out2_
                d_6_stepsUsed_ = out3_
                d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed_)
                if d_5_stoppedOnEos_:
                    generated = d_3_genOut_
                    cost = d_1_steps_
                elif True:
                    generated = d_3_genOut_
                    if d_4_stoppedOnOpenSpan_:
                        d_7_enterGenerated_: _dafny.Seq
                        d_8_enterInside_: bool
                        d_9_enterCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_7_enterGenerated_ = out4_
                        d_8_enterInside_ = out5_
                        d_9_enterCurrent_ = out6_
                        generated = d_7_enterGenerated_
                        insideConstrainedOut = d_8_enterInside_
                        currentConstrainedOut = d_9_enterCurrent_
                    elif (d_1_steps_) < (maxSteps):
                        d_10_openGenerated_: _dafny.Seq
                        d_11_openInside_: bool
                        d_12_openCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_10_openGenerated_ = out7_
                        d_11_openInside_ = out8_
                        d_12_openCurrent_ = out9_
                        generated = d_10_openGenerated_
                        insideConstrainedOut = d_11_openInside_
                        currentConstrainedOut = d_12_openCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    with _dafny.label("1_0_2_0"):
                        while (((d_1_steps_) < (maxSteps)) and (insideConstrainedOut)) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                            with _dafny.c_label("1_0_2_0"):
                                d_13_constrainedPrompt_: _dafny.Seq
                                d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_14_next_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_14_next_ = out10_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_14_next_) == (eosToken):
                                    raise _dafny.Break("1_0_2_0")
                                elif True:
                                    d_15_appendedGenerated_: _dafny.Seq
                                    d_16_appendedInside_: bool
                                    d_17_appendedCurrent_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                    d_15_appendedGenerated_ = out11_
                                    d_16_appendedInside_ = out12_
                                    d_17_appendedCurrent_ = out13_
                                    generated = d_15_appendedGenerated_
                                    insideConstrainedOut = d_16_appendedInside_
                                    currentConstrainedOut = d_17_appendedCurrent_
                                pass
                        pass
                    if (((d_1_steps_) < (maxSteps)) and (insideConstrainedOut)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                        d_18_closedGenerated_: _dafny.Seq
                        d_19_closedInside_: bool
                        d_20_closedCurrent_: _dafny.Seq
                        out14_: _dafny.Seq
                        out15_: bool
                        out16_: _dafny.Seq
                        out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_18_closedGenerated_ = out14_
                        d_19_closedInside_ = out15_
                        d_20_closedCurrent_ = out16_
                        generated = d_18_closedGenerated_
                        insideConstrainedOut = d_19_closedInside_
                        currentConstrainedOut = d_20_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    cost = d_1_steps_
            elif insideConstrainedOut:
                with _dafny.label("1_1_0_0"):
                    while (((d_1_steps_) < (maxSteps)) and (insideConstrainedOut)) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                        with _dafny.c_label("1_1_0_0"):
                            d_21_constrainedPrompt_: _dafny.Seq
                            d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_22_next_: _dafny.Seq
                            out17_: _dafny.Seq
                            out17_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_22_next_ = out17_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_22_next_) == (eosToken):
                                raise _dafny.Break("1_1_0_0")
                            elif True:
                                d_23_appendedGenerated_: _dafny.Seq
                                d_24_appendedInside_: bool
                                d_25_appendedCurrent_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_next_)
                                d_23_appendedGenerated_ = out18_
                                d_24_appendedInside_ = out19_
                                d_25_appendedCurrent_ = out20_
                                generated = d_23_appendedGenerated_
                                insideConstrainedOut = d_24_appendedInside_
                                currentConstrainedOut = d_25_appendedCurrent_
                            pass
                    pass
                if (((d_1_steps_) < (maxSteps)) and (insideConstrainedOut)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                    d_26_closedGenerated_: _dafny.Seq
                    d_27_closedInside_: bool
                    d_28_closedCurrent_: _dafny.Seq
                    out21_: _dafny.Seq
                    out22_: bool
                    out23_: _dafny.Seq
                    out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_26_closedGenerated_ = out21_
                    d_27_closedInside_ = out22_
                    d_28_closedCurrent_ = out23_
                    generated = d_26_closedGenerated_
                    insideConstrainedOut = d_27_closedInside_
                    currentConstrainedOut = d_28_closedCurrent_
                    d_1_steps_ = (d_1_steps_) + (1)
                cost = d_1_steps_
            elif True:
                cost = 0
        return generated, insideConstrainedOut, currentConstrainedOut, cost

