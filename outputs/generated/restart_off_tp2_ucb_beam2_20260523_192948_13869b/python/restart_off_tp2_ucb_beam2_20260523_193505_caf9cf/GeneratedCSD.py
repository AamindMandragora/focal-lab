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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the problem step by step using the variable names from the question. Wrap EVERY intermediate computation and the FINAL answer inside << and >>. Use Python integer division // (not /) when the result must be a whole number. Always end with one explicit line of the form: The answer is <<FORMULA>>. Make sure FORMULA contains the COMPLETE expression with ALL factors, not just a partial step. Example format: 'There are <<a*b>> items per day. Over <<n>> days the total is <<a*b*n>>. The answer is <<a*b*n>>.'")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanCap_: int
        d_2_spanCap_ = 32
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_remaining_: int
                        d_3_remaining_ = (maxSteps) - (d_1_steps_)
                        d_4_chunkBudget_: int
                        if (d_3_remaining_) > (60):
                            d_4_chunkBudget_ = 60
                        elif True:
                            d_4_chunkBudget_ = d_3_remaining_
                        d_5_chunkedG_: _dafny.Seq
                        d_6_stoppedOpen_: bool
                        d_7_stoppedEos_: bool
                        d_8_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_5_chunkedG_ = out0_
                        d_6_stoppedOpen_ = out1_
                        d_7_stoppedEos_ = out2_
                        d_8_stepsUsed_ = out3_
                        generated = d_5_chunkedG_
                        d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                        if d_7_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_6_stoppedOpen_:
                            d_9_enteredGenerated_: _dafny.Seq
                            d_10_enteredInside_: bool
                            d_11_enteredCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_9_enteredGenerated_ = out4_
                            d_10_enteredInside_ = out5_
                            d_11_enteredCurrent_ = out6_
                            generated = d_9_enteredGenerated_
                            insideConstrainedOut = d_10_enteredInside_
                            currentConstrainedOut = d_11_enteredCurrent_
                        elif (d_8_stepsUsed_) == (0):
                            d_12_next_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_12_next_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_12_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                                if (d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_13_closedGenerated_: _dafny.Seq
                        d_14_closedInside_: bool
                        d_15_closedCurrent_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_13_closedGenerated_ = out8_
                        d_14_closedInside_ = out9_
                        d_15_closedCurrent_ = out10_
                        generated = d_13_closedGenerated_
                        insideConstrainedOut = d_14_closedInside_
                        currentConstrainedOut = d_15_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_2_spanCap_):
                        d_16_rolledGenerated_: _dafny.Seq
                        d_17_rolledCurrent_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: _dafny.Seq
                        out11_, out12_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_16_rolledGenerated_ = out11_
                        d_17_rolledCurrent_ = out12_
                        generated = d_16_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_17_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_18_stablePrefix_: _dafny.Seq
                        d_18_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_19_constrainedPrompt_: _dafny.Seq
                        d_19_constrainedPrompt_ = (prompt) + (d_18_stablePrefix_)
                        d_20_validCount_: int
                        out13_: int
                        out13_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_20_validCount_ = out13_
                        if (d_20_validCount_) <= (6):
                            d_21_next_: _dafny.Seq
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_21_next_ = out14_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_21_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_22_appendedGenerated_: _dafny.Seq
                                d_23_appendedInside_: bool
                                d_24_appendedCurrent_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                d_22_appendedGenerated_ = out15_
                                d_23_appendedInside_ = out16_
                                d_24_appendedCurrent_ = out17_
                                generated = d_22_appendedGenerated_
                                insideConstrainedOut = d_23_appendedInside_
                                currentConstrainedOut = d_24_appendedCurrent_
                        elif True:
                            d_25_remaining_: int
                            d_25_remaining_ = (maxSteps) - (d_1_steps_)
                            d_26_symbolBudget_: int
                            if (d_25_remaining_) > (8):
                                d_26_symbolBudget_ = 8
                            elif True:
                                d_26_symbolBudget_ = d_25_remaining_
                            if (d_26_symbolBudget_) == (0):
                                d_26_symbolBudget_ = 1
                            d_27_symbolGenerated_: _dafny.Seq
                            d_28_symbolOut_: _dafny.Seq
                            d_29_hitEos_: bool
                            d_30_stepsUsed_: int
                            out18_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: int
                            out18_, out19_, out20_, out21_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_19_constrainedPrompt_, generated, currentConstrainedOut, d_26_symbolBudget_, eosToken)
                            d_27_symbolGenerated_ = out18_
                            d_28_symbolOut_ = out19_
                            d_29_hitEos_ = out20_
                            d_30_stepsUsed_ = out21_
                            generated = d_27_symbolGenerated_
                            currentConstrainedOut = d_28_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_30_stepsUsed_)
                            if d_29_hitEos_:
                                raise _dafny.Break("0")
                            if (d_30_stepsUsed_) == (0):
                                d_31_next_: _dafny.Seq
                                out22_: _dafny.Seq
                                out22_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_31_next_ = out22_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_31_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_32_appendedGenerated_: _dafny.Seq
                                    d_33_appendedInside_: bool
                                    d_34_appendedCurrent_: _dafny.Seq
                                    out23_: _dafny.Seq
                                    out24_: bool
                                    out25_: _dafny.Seq
                                    out23_, out24_, out25_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_31_next_)
                                    d_32_appendedGenerated_ = out23_
                                    d_33_appendedInside_ = out24_
                                    d_34_appendedCurrent_ = out25_
                                    generated = d_32_appendedGenerated_
                                    insideConstrainedOut = d_33_appendedInside_
                                    currentConstrainedOut = d_34_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

