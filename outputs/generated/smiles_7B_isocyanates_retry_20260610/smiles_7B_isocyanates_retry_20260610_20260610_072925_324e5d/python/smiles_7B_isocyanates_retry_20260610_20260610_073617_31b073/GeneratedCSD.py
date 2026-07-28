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
            (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You are a chemistry assistant. Think step by step about the molecular structure of the given chemical name, then write the canonical SMILES string between << and >>. First write your reasoning, then write <<SMILES_HERE>>. The SMILES must be the complete correct structure.")))
            d_1_steps_: int
            d_1_steps_ = 0
            if not(insideConstrainedOut):
                d_2_reservedForConstrained_: int
                d_2_reservedForConstrained_ = 80
                d_3_maxChunk_: int = int(0)
                if (maxSteps) > ((d_2_reservedForConstrained_) + (2)):
                    d_3_maxChunk_ = ((maxSteps) - (d_2_reservedForConstrained_)) - (1)
                elif (maxSteps) > (2):
                    d_3_maxChunk_ = (maxSteps) - (2)
                elif True:
                    d_3_maxChunk_ = 1
                d_4_genOut_: _dafny.Seq
                d_5_stoppedOnOpenSpan_: bool
                d_6_stoppedOnEos_: bool
                d_7_stepsUsed_: int
                out0_: _dafny.Seq
                out1_: bool
                out2_: bool
                out3_: int
                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_maxChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                d_4_genOut_ = out0_
                d_5_stoppedOnOpenSpan_ = out1_
                d_6_stoppedOnEos_ = out2_
                d_7_stepsUsed_ = out3_
                d_1_steps_ = (d_1_steps_) + (d_7_stepsUsed_)
                generated = d_4_genOut_
                if d_6_stoppedOnEos_:
                    cost = d_1_steps_
                elif True:
                    if d_5_stoppedOnOpenSpan_:
                        d_8_enterGenerated_: _dafny.Seq
                        d_9_enterInside_: bool
                        d_10_enterCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                        d_8_enterGenerated_ = out4_
                        d_9_enterInside_ = out5_
                        d_10_enterCurrent_ = out6_
                        generated = d_8_enterGenerated_
                        insideConstrainedOut = d_9_enterInside_
                        currentConstrainedOut = d_10_enterCurrent_
                    elif True:
                        if (d_1_steps_) < (maxSteps):
                            d_11_openGenerated_: _dafny.Seq
                            d_12_openInside_: bool
                            d_13_openCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_11_openGenerated_ = out7_
                            d_12_openInside_ = out8_
                            d_13_openCurrent_ = out9_
                            generated = d_11_openGenerated_
                            insideConstrainedOut = d_12_openInside_
                            currentConstrainedOut = d_13_openCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                    with _dafny.label("1_0_3_0"):
                        while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                            with _dafny.c_label("1_0_3_0"):
                                d_14_cg_: _dafny.Seq
                                d_15_ci_: bool
                                d_16_cc_: _dafny.Seq
                                d_17_closed_: bool
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out13_: bool
                                out10_, out11_, out12_, out13_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                d_14_cg_ = out10_
                                d_15_ci_ = out11_
                                d_16_cc_ = out12_
                                d_17_closed_ = out13_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if d_17_closed_:
                                    generated = d_14_cg_
                                    insideConstrainedOut = d_15_ci_
                                    currentConstrainedOut = d_16_cc_
                                elif True:
                                    if (d_1_steps_) < (maxSteps):
                                        d_18_constrainedPrompt_: _dafny.Seq
                                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                        d_19_next_: _dafny.Seq
                                        out14_: _dafny.Seq
                                        out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                        d_19_next_ = out14_
                                        d_1_steps_ = (d_1_steps_) + (1)
                                        if (d_19_next_) == (eosToken):
                                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                                if (d_1_steps_) < (maxSteps):
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
                                            raise _dafny.Break("1_0_3_0")
                                        elif True:
                                            d_23_appendedGenerated_: _dafny.Seq
                                            d_24_appendedInside_: bool
                                            d_25_appendedCurrent_: _dafny.Seq
                                            out18_: _dafny.Seq
                                            out19_: bool
                                            out20_: _dafny.Seq
                                            out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                            d_23_appendedGenerated_ = out18_
                                            d_24_appendedInside_ = out19_
                                            d_25_appendedCurrent_ = out20_
                                            generated = d_23_appendedGenerated_
                                            insideConstrainedOut = d_24_appendedInside_
                                            currentConstrainedOut = d_25_appendedCurrent_
                                pass
                        pass
                    cost = d_1_steps_
            elif True:
                with _dafny.label("1_1_0"):
                    while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                        with _dafny.c_label("1_1_0"):
                            d_26_cg_: _dafny.Seq
                            d_27_ci_: bool
                            d_28_cc_: _dafny.Seq
                            d_29_closed_: bool
                            out21_: _dafny.Seq
                            out22_: bool
                            out23_: _dafny.Seq
                            out24_: bool
                            out21_, out22_, out23_, out24_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_26_cg_ = out21_
                            d_27_ci_ = out22_
                            d_28_cc_ = out23_
                            d_29_closed_ = out24_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if d_29_closed_:
                                generated = d_26_cg_
                                insideConstrainedOut = d_27_ci_
                                currentConstrainedOut = d_28_cc_
                            elif True:
                                if (d_1_steps_) < (maxSteps):
                                    d_30_constrainedPrompt_: _dafny.Seq
                                    d_30_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_31_next_: _dafny.Seq
                                    out25_: _dafny.Seq
                                    out25_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_30_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                    d_31_next_ = out25_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_31_next_) == (eosToken):
                                        raise _dafny.Break("1_1_0")
                                    elif True:
                                        d_32_appendedGenerated_: _dafny.Seq
                                        d_33_appendedInside_: bool
                                        d_34_appendedCurrent_: _dafny.Seq
                                        out26_: _dafny.Seq
                                        out27_: bool
                                        out28_: _dafny.Seq
                                        out26_, out27_, out28_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_31_next_)
                                        d_32_appendedGenerated_ = out26_
                                        d_33_appendedInside_ = out27_
                                        d_34_appendedCurrent_ = out28_
                                        generated = d_32_appendedGenerated_
                                        insideConstrainedOut = d_33_appendedInside_
                                        currentConstrainedOut = d_34_appendedCurrent_
                            pass
                    pass
                cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

