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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a valid SMILES string for the requested molecular class. Prefer a complete molecule rather than tiny fragments, and keep any constrained-span content syntactically valid.")))
        (d_0_helpers_).SetNonDeterministic(lm, False)
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_3_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                            if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_4_closedGenerated_: _dafny.Seq
                        d_5_closedInside_: bool
                        d_6_closedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_4_closedGenerated_ = out1_
                        d_5_closedInside_ = out2_
                        d_6_closedCurrent_ = out3_
                        generated = d_4_closedGenerated_
                        insideConstrainedOut = d_5_closedInside_
                        currentConstrainedOut = d_6_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_7_stablePrefix_: _dafny.Seq
                        d_7_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_8_constrainedPrompt_: _dafny.Seq
                        d_8_constrainedPrompt_ = (prompt) + (d_7_stablePrefix_)
                        d_9_validCount_: int
                        out4_: int
                        out4_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_9_validCount_ = out4_
                        if (len(currentConstrainedOut)) < (2):
                            d_10_nextEarly_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), eosToken)
                            d_10_nextEarly_ = out5_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_10_nextEarly_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_11_appendedGenerated_: _dafny.Seq
                                d_12_appendedInside_: bool
                                d_13_appendedCurrent_: _dafny.Seq
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: _dafny.Seq
                                out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_nextEarly_)
                                d_11_appendedGenerated_ = out6_
                                d_12_appendedInside_ = out7_
                                d_13_appendedCurrent_ = out8_
                                generated = d_11_appendedGenerated_
                                insideConstrainedOut = d_12_appendedInside_
                                currentConstrainedOut = d_13_appendedCurrent_
                        elif (d_9_validCount_) <= (d_2_narrowThreshold_):
                            d_14_nextNarrow_: _dafny.Seq
                            d_15_wasConstrained_: bool
                            out9_: _dafny.Seq
                            out10_: bool
                            out9_, out10_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_14_nextNarrow_ = out9_
                            d_15_wasConstrained_ = out10_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_14_nextNarrow_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_16_appendedGenerated2_: _dafny.Seq
                                d_17_appendedInside2_: bool
                                d_18_appendedCurrent2_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_nextNarrow_)
                                d_16_appendedGenerated2_ = out11_
                                d_17_appendedInside2_ = out12_
                                d_18_appendedCurrent2_ = out13_
                                generated = d_16_appendedGenerated2_
                                insideConstrainedOut = d_17_appendedInside2_
                                currentConstrainedOut = d_18_appendedCurrent2_
                        elif True:
                            d_19_remaining_: int
                            d_19_remaining_ = (maxSteps) - (d_1_steps_)
                            d_20_symbolBudget_: int
                            if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_19_remaining_)):
                                d_20_symbolBudget_ = d_19_remaining_
                            elif True:
                                d_20_symbolBudget_ = stepTokenBudget
                            d_21_symbolGenerated_: _dafny.Seq
                            d_22_symbolOut_: _dafny.Seq
                            d_23_hitEos_: bool
                            d_24_stepsUsed_: int
                            out14_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: int
                            out14_, out15_, out16_, out17_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_8_constrainedPrompt_, generated, currentConstrainedOut, d_20_symbolBudget_, eosToken)
                            d_21_symbolGenerated_ = out14_
                            d_22_symbolOut_ = out15_
                            d_23_hitEos_ = out16_
                            d_24_stepsUsed_ = out17_
                            generated = d_21_symbolGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_22_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_24_stepsUsed_)
                            if d_23_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

