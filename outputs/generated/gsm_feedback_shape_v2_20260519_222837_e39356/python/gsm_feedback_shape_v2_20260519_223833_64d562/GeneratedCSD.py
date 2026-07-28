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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write each arithmetic computation inside visible delimiters << and >>, and close each computation span immediately after the computation.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_2_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_2_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                            if (d_2_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_3_observedGenerated_: _dafny.Seq
                                d_4_observedInside_: bool
                                d_5_observedCurrent_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_3_observedGenerated_ = out1_
                                d_4_observedInside_ = out2_
                                d_5_observedCurrent_ = out3_
                                generated = d_3_observedGenerated_
                                insideConstrainedOut = d_4_observedInside_
                                currentConstrainedOut = d_5_observedCurrent_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_6_closedGenerated_: _dafny.Seq
                        d_7_closedInside_: bool
                        d_8_closedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_6_closedGenerated_ = out4_
                        d_7_closedInside_ = out5_
                        d_8_closedCurrent_ = out6_
                        generated = d_6_closedGenerated_
                        insideConstrainedOut = d_7_closedInside_
                        currentConstrainedOut = d_8_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_9_stablePrefix_: _dafny.Seq
                        d_9_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_10_constrainedPrompt_: _dafny.Seq
                        d_10_constrainedPrompt_ = (prompt) + (d_9_stablePrefix_)
                        d_11_remaining_: int
                        d_11_remaining_ = (maxSteps) - (d_1_steps_)
                        d_12_symbolBudget_: int
                        if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_11_remaining_)):
                            d_12_symbolBudget_ = d_11_remaining_
                        elif True:
                            d_12_symbolBudget_ = stepTokenBudget
                        d_13_symbolGenerated_: _dafny.Seq
                        d_14_symbolOut_: _dafny.Seq
                        d_15_hitEos_: bool
                        d_16_stepsUsed_: int
                        out7_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: int
                        out7_, out8_, out9_, out10_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_10_constrainedPrompt_, generated, currentConstrainedOut, d_12_symbolBudget_, eosToken)
                        d_13_symbolGenerated_ = out7_
                        d_14_symbolOut_ = out8_
                        d_15_hitEos_ = out9_
                        d_16_stepsUsed_ = out10_
                        generated = d_13_symbolGenerated_
                        currentConstrainedOut = d_14_symbolOut_
                        d_1_steps_ = (d_1_steps_) + (d_16_stepsUsed_)
                        if d_15_hitEos_:
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

