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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output the answer as a visible constrained span containing a valid SMILES string for the requested molecular class. Prefer a single complete SMILES and keep every constrained prefix parser-valid.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_openedGenerated_: _dafny.Seq
                        d_3_openedInside_: bool
                        d_4_openedCurrent_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_2_openedGenerated_ = out0_
                        d_3_openedInside_ = out1_
                        d_4_openedCurrent_ = out2_
                        generated = d_2_openedGenerated_
                        insideConstrainedOut = d_3_openedInside_
                        currentConstrainedOut = d_4_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_5_closedGenerated_: _dafny.Seq
                        d_6_closedInside_: bool
                        d_7_closedCurrent_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_5_closedGenerated_ = out3_
                        d_6_closedInside_ = out4_
                        d_7_closedCurrent_ = out5_
                        generated = d_5_closedGenerated_
                        insideConstrainedOut = d_6_closedInside_
                        currentConstrainedOut = d_7_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_8_stablePrefix_: _dafny.Seq
                        d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_9_constrainedPrompt_: _dafny.Seq
                        d_9_constrainedPrompt_ = (prompt) + (d_8_stablePrefix_)
                        d_10_validCount_: int
                        out6_: int
                        out6_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_10_validCount_ = out6_
                        if (d_10_validCount_) <= (4):
                            d_11_nextTight_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_11_nextTight_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_11_nextTight_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_12_appendedGenerated_: _dafny.Seq
                                d_13_appendedInside_: bool
                                d_14_appendedCurrent_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_nextTight_)
                                d_12_appendedGenerated_ = out8_
                                d_13_appendedInside_ = out9_
                                d_14_appendedCurrent_ = out10_
                                generated = d_12_appendedGenerated_
                                insideConstrainedOut = d_13_appendedInside_
                                currentConstrainedOut = d_14_appendedCurrent_
                        elif (d_10_validCount_) <= (16):
                            d_15_nextMid_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 16, eosToken)
                            d_15_nextMid_ = out11_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_15_nextMid_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_16_appendedGenerated2_: _dafny.Seq
                                d_17_appendedInside2_: bool
                                d_18_appendedCurrent2_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_nextMid_)
                                d_16_appendedGenerated2_ = out12_
                                d_17_appendedInside2_ = out13_
                                d_18_appendedCurrent2_ = out14_
                                generated = d_16_appendedGenerated2_
                                insideConstrainedOut = d_17_appendedInside2_
                                currentConstrainedOut = d_18_appendedCurrent2_
                        elif (len(validTokenGroups)) > (0):
                            d_19_nextGroup_: _dafny.Seq
                            out15_: _dafny.Seq
                            out15_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                            d_19_nextGroup_ = out15_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_19_nextGroup_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_20_appendedGenerated3_: _dafny.Seq
                                d_21_appendedInside3_: bool
                                d_22_appendedCurrent3_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_nextGroup_)
                                d_20_appendedGenerated3_ = out16_
                                d_21_appendedInside3_ = out17_
                                d_22_appendedCurrent3_ = out18_
                                generated = d_20_appendedGenerated3_
                                insideConstrainedOut = d_21_appendedInside3_
                                currentConstrainedOut = d_22_appendedCurrent3_
                        elif True:
                            d_23_remaining_: int
                            d_23_remaining_ = (maxSteps) - (d_1_steps_)
                            d_24_symbolBudget_: int
                            if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_23_remaining_)):
                                d_24_symbolBudget_ = d_23_remaining_
                            elif True:
                                d_24_symbolBudget_ = stepTokenBudget
                            d_25_symbolGenerated_: _dafny.Seq
                            d_26_symbolCurrent_: _dafny.Seq
                            d_27_hitEos_: bool
                            d_28_stepsUsed_: int
                            out19_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: bool
                            out22_: int
                            out19_, out20_, out21_, out22_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_9_constrainedPrompt_, generated, currentConstrainedOut, d_24_symbolBudget_, eosToken)
                            d_25_symbolGenerated_ = out19_
                            d_26_symbolCurrent_ = out20_
                            d_27_hitEos_ = out21_
                            d_28_stepsUsed_ = out22_
                            generated = d_25_symbolGenerated_
                            currentConstrainedOut = d_26_symbolCurrent_
                            d_1_steps_ = (d_1_steps_) + (d_28_stepsUsed_)
                            if d_27_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

