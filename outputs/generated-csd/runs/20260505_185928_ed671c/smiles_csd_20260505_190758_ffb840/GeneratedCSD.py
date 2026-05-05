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
        d_3_recentAfterAnswer_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))
        d_3_recentAfterAnswer_ = out0_
        d_4_narrowThreshold_: int
        d_4_narrowThreshold_ = 12
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        out1_: _dafny.Seq
                        out1_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))
                        d_3_recentAfterAnswer_ = out1_
                        if (not(d_2_chemistryCueSeen_)) and ((len(d_3_recentAfterAnswer_)) > (0)):
                            d_2_chemistryCueSeen_ = True
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
                                    d_9_observedGenerated_: _dafny.Seq
                                    d_10_observedInside_: bool
                                    d_11_observedCurrent_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out7_: bool
                                    out8_: _dafny.Seq
                                    out6_, out7_, out8_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_9_observedGenerated_ = out6_
                                    d_10_observedInside_ = out7_
                                    d_11_observedCurrent_ = out8_
                                    generated = d_9_observedGenerated_
                                    insideConstrainedOut = d_10_observedInside_
                                    currentConstrainedOut = d_11_observedCurrent_
                                    d_2_chemistryCueSeen_ = False
                                elif (((((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))) or ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SMILES"))))) or ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "smiles"))))) or ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "molecule"))))) or ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "structure")))):
                                    d_2_chemistryCueSeen_ = True
                    elif True:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
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
                            raise _dafny.Break("0")
                        elif True:
                            d_15_stablePrefix_: _dafny.Seq
                            d_15_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_16_constrainedPrompt_: _dafny.Seq
                            d_16_constrainedPrompt_ = (prompt) + (d_15_stablePrefix_)
                            d_17_validCount_: int
                            out12_: int
                            out12_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_17_validCount_ = out12_
                            if (d_17_validCount_) <= (d_4_narrowThreshold_):
                                d_18_next_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_4_narrowThreshold_, eosToken)
                                d_18_next_ = out13_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_18_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_19_appendedGenerated_: _dafny.Seq
                                    d_20_appendedInside_: bool
                                    d_21_appendedCurrent_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                                    d_19_appendedGenerated_ = out14_
                                    d_20_appendedInside_ = out15_
                                    d_21_appendedCurrent_ = out16_
                                    generated = d_19_appendedGenerated_
                                    insideConstrainedOut = d_20_appendedInside_
                                    currentConstrainedOut = d_21_appendedCurrent_
                            elif True:
                                d_22_remaining_: int
                                d_22_remaining_ = (maxSteps) - (d_1_steps_)
                                d_23_symbolBudget_: int
                                if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_22_remaining_)):
                                    d_23_symbolBudget_ = d_22_remaining_
                                elif True:
                                    d_23_symbolBudget_ = stepTokenBudget
                                d_24_symbolGenerated_: _dafny.Seq
                                d_25_symbolCurrent_: _dafny.Seq
                                d_26_hitEos_: bool
                                d_27_stepsUsed_: int
                                out17_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: int
                                out17_, out18_, out19_, out20_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_16_constrainedPrompt_, generated, currentConstrainedOut, d_23_symbolBudget_, eosToken)
                                d_24_symbolGenerated_ = out17_
                                d_25_symbolCurrent_ = out18_
                                d_26_hitEos_ = out19_
                                d_27_stepsUsed_ = out20_
                                generated = d_24_symbolGenerated_
                                currentConstrainedOut = d_25_symbolCurrent_
                                d_1_steps_ = (d_1_steps_) + (d_27_stepsUsed_)
                                if d_26_hitEos_:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

