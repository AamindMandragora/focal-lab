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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Produce exactly one valid SMILES string for the requested molecular class. Use a constrained span only for the final SMILES string, and do not open it until ready to emit the full molecule.")))
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
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_3_closedGenerated_: _dafny.Seq
                        d_4_closedInside_: bool
                        d_5_closedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_3_closedGenerated_ = out1_
                        d_4_closedInside_ = out2_
                        d_5_closedCurrent_ = out3_
                        generated = d_3_closedGenerated_
                        insideConstrainedOut = d_4_closedInside_
                        currentConstrainedOut = d_5_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_6_constrainedPrompt_: _dafny.Seq
                        d_6_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_7_remaining_: int
                        d_7_remaining_ = (maxSteps) - (d_1_steps_)
                        d_8_symbolBudget_: int
                        if (d_7_remaining_) == (0):
                            d_8_symbolBudget_ = 0
                        elif ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_7_remaining_)):
                            d_8_symbolBudget_ = d_7_remaining_
                        elif True:
                            d_8_symbolBudget_ = stepTokenBudget
                        d_9_symbolGenerated_: _dafny.Seq
                        d_10_symbolOut_: _dafny.Seq
                        d_11_hitEos_: bool
                        d_12_stepsUsed_: int
                        out4_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: int
                        out4_, out5_, out6_, out7_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_6_constrainedPrompt_, generated, currentConstrainedOut, d_8_symbolBudget_, eosToken)
                        d_9_symbolGenerated_ = out4_
                        d_10_symbolOut_ = out5_
                        d_11_hitEos_ = out6_
                        d_12_stepsUsed_ = out7_
                        generated = d_9_symbolGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_10_symbolOut_
                        d_1_steps_ = (d_1_steps_) + (d_12_stepsUsed_)
                        if d_11_hitEos_:
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

