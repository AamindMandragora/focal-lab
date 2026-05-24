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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Write each arithmetic computation inside << >> delimiters.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_sawAnyOpen_: bool
        d_2_sawAnyOpen_ = insideConstrained
        d_3_forcedFirstOpen_: bool
        d_3_forcedFirstOpen_ = False
        d_4_preludeDone_: bool
        d_4_preludeDone_ = False
        if not(d_2_sawAnyOpen_):
            d_5_initialOpenCount_: int
            out0_: int
            out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generatedPrefix, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
            d_5_initialOpenCount_ = out0_
            if (d_5_initialOpenCount_) > (0):
                d_2_sawAnyOpen_ = True
                d_3_forcedFirstOpen_ = True
                d_4_preludeDone_ = True
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if not(d_4_preludeDone_):
                            d_6_remainingPrelude_: int
                            d_6_remainingPrelude_ = (maxSteps) - (d_1_steps_)
                            d_7_preludeBudget_: int
                            if (d_6_remainingPrelude_) > (8):
                                d_7_preludeBudget_ = 8
                            elif True:
                                d_7_preludeBudget_ = d_6_remainingPrelude_
                            d_8_chunkedG_: _dafny.Seq
                            d_9_stoppedOpen_: bool
                            d_10_stoppedEos_: bool
                            d_11_stepsUsed_: int
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: bool
                            out4_: int
                            out1_, out2_, out3_, out4_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_7_preludeBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_8_chunkedG_ = out1_
                            d_9_stoppedOpen_ = out2_
                            d_10_stoppedEos_ = out3_
                            d_11_stepsUsed_ = out4_
                            generated = d_8_chunkedG_
                            d_1_steps_ = (d_1_steps_) + (d_11_stepsUsed_)
                            d_4_preludeDone_ = True
                            if d_10_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_9_stoppedOpen_:
                                d_12_enteredGenerated_: _dafny.Seq
                                d_13_enteredInside_: bool
                                d_14_enteredCurrent_: _dafny.Seq
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: _dafny.Seq
                                out5_, out6_, out7_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_12_enteredGenerated_ = out5_
                                d_13_enteredInside_ = out6_
                                d_14_enteredCurrent_ = out7_
                                generated = d_12_enteredGenerated_
                                insideConstrainedOut = d_13_enteredInside_
                                currentConstrainedOut = d_14_enteredCurrent_
                                d_2_sawAnyOpen_ = True
                        elif not(d_3_forcedFirstOpen_):
                            d_15_openedGenerated_: _dafny.Seq
                            d_16_openedInside_: bool
                            d_17_openedCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_15_openedGenerated_ = out8_
                            d_16_openedInside_ = out9_
                            d_17_openedCurrent_ = out10_
                            generated = d_15_openedGenerated_
                            insideConstrainedOut = d_16_openedInside_
                            currentConstrainedOut = d_17_openedCurrent_
                            d_3_forcedFirstOpen_ = True
                            d_2_sawAnyOpen_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_18_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_18_next_ = out11_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_18_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_18_next_]))
                                if (d_18_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_19_enteredGenerated2_: _dafny.Seq
                                    d_20_enteredInside2_: bool
                                    d_21_enteredCurrent2_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_19_enteredGenerated2_ = out12_
                                    d_20_enteredInside2_ = out13_
                                    d_21_enteredCurrent2_ = out14_
                                    generated = d_19_enteredGenerated2_
                                    insideConstrainedOut = d_20_enteredInside2_
                                    currentConstrainedOut = d_21_enteredCurrent2_
                                    d_2_sawAnyOpen_ = True
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
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_25_stablePrefix_: _dafny.Seq
                        d_25_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_26_constrainedPrompt_: _dafny.Seq
                        d_26_constrainedPrompt_ = (prompt) + (d_25_stablePrefix_)
                        d_27_remaining_: int
                        d_27_remaining_ = (maxSteps) - (d_1_steps_)
                        d_28_symbolBudget_: int
                        if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_27_remaining_)):
                            d_28_symbolBudget_ = d_27_remaining_
                        elif True:
                            d_28_symbolBudget_ = stepTokenBudget
                        d_29_symbolGenerated_: _dafny.Seq
                        d_30_symbolOut_: _dafny.Seq
                        d_31_hitEos_: bool
                        d_32_stepsUsed2_: int
                        out18_: _dafny.Seq
                        out19_: _dafny.Seq
                        out20_: bool
                        out21_: int
                        out18_, out19_, out20_, out21_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_26_constrainedPrompt_, generated, currentConstrainedOut, d_28_symbolBudget_, eosToken)
                        d_29_symbolGenerated_ = out18_
                        d_30_symbolOut_ = out19_
                        d_31_hitEos_ = out20_
                        d_32_stepsUsed2_ = out21_
                        generated = d_29_symbolGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_30_symbolOut_
                        d_1_steps_ = (d_1_steps_) + (d_32_stepsUsed2_)
                        if d_31_hitEos_:
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

