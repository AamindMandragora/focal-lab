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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "When producing a constrained span, emit a valid SMILES string for the requested molecular class. Prefer chemically plausible continuations, but preserve parser-validity at every committed step.")))
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
                        d_9_remaining_: int
                        d_9_remaining_ = (maxSteps) - (d_1_steps_)
                        d_10_symbolBudget_: int
                        if (stepTokenBudget) == (0):
                            d_10_symbolBudget_ = 1
                        elif (stepTokenBudget) > (d_9_remaining_):
                            d_10_symbolBudget_ = d_9_remaining_
                        elif True:
                            d_10_symbolBudget_ = stepTokenBudget
                        d_11_validCount_: int
                        out4_: int
                        out4_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_11_validCount_ = out4_
                        if (((d_11_validCount_) <= (d_2_narrowThreshold_)) or ((len(currentConstrainedOut)) < (2))) or ((d_10_symbolBudget_) == (1)):
                            d_12_next_: _dafny.Seq
                            d_12_next_ = eosToken
                            if (d_11_validCount_) <= (d_2_narrowThreshold_):
                                out5_: _dafny.Seq
                                out5_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_2_narrowThreshold_, eosToken)
                                d_12_next_ = out5_
                            elif True:
                                out6_: _dafny.Seq
                                out6_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('3e0'), eosToken)
                                d_12_next_ = out6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_12_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_13_appendedGenerated_: _dafny.Seq
                                d_14_appendedInside_: bool
                                d_15_appendedCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                                d_13_appendedGenerated_ = out7_
                                d_14_appendedInside_ = out8_
                                d_15_appendedCurrent_ = out9_
                                generated = d_13_appendedGenerated_
                                insideConstrainedOut = d_14_appendedInside_
                                currentConstrainedOut = d_15_appendedCurrent_
                        elif True:
                            d_16_symbolGenerated_: _dafny.Seq
                            d_17_symbolOut_: _dafny.Seq
                            d_18_hitEos_: bool
                            d_19_stepsUsed_: int
                            out10_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: int
                            out10_, out11_, out12_, out13_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_8_constrainedPrompt_, generated, currentConstrainedOut, d_10_symbolBudget_, eosToken)
                            d_16_symbolGenerated_ = out10_
                            d_17_symbolOut_ = out11_
                            d_18_hitEos_ = out12_
                            d_19_stepsUsed_ = out13_
                            generated = d_16_symbolGenerated_
                            insideConstrainedOut = True
                            currentConstrainedOut = d_17_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_19_stepsUsed_)
                            if d_18_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

