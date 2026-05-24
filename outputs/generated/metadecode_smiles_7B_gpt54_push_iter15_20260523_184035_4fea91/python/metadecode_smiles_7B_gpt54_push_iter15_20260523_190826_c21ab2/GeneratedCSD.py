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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Produce exactly one valid SMILES string for the requested molecular class. If a constrained span is used, put the full final SMILES inside it and avoid extra commentary.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_forcedOpenUsed_: bool
        d_2_forcedOpenUsed_ = insideConstrained
        d_3_openAfter_: int
        d_3_openAfter_ = 24
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_2_forcedOpenUsed_)) and ((d_1_steps_) >= (d_3_openAfter_)):
                            d_4_openedGenerated_: _dafny.Seq
                            d_5_openedInside_: bool
                            d_6_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_openedGenerated_ = out0_
                            d_5_openedInside_ = out1_
                            d_6_openedCurrent_ = out2_
                            generated = d_4_openedGenerated_
                            insideConstrainedOut = d_5_openedInside_
                            currentConstrainedOut = d_6_openedCurrent_
                            d_2_forcedOpenUsed_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_7_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                                if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_forcedOpenUsed_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_8_closedGenerated_: _dafny.Seq
                        d_9_closedInside_: bool
                        d_10_closedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_8_closedGenerated_ = out4_
                        d_9_closedInside_ = out5_
                        d_10_closedCurrent_ = out6_
                        generated = d_8_closedGenerated_
                        insideConstrainedOut = d_9_closedInside_
                        currentConstrainedOut = d_10_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_11_constrainedPrompt_: _dafny.Seq
                        d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_12_nextInside_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_12_nextInside_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_12_nextInside_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_13_appendedGenerated_: _dafny.Seq
                            d_14_appendedInside_: bool
                            d_15_appendedCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_nextInside_)
                            d_13_appendedGenerated_ = out8_
                            d_14_appendedInside_ = out9_
                            d_15_appendedCurrent_ = out10_
                            generated = d_13_appendedGenerated_
                            insideConstrainedOut = d_14_appendedInside_
                            currentConstrainedOut = d_15_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

