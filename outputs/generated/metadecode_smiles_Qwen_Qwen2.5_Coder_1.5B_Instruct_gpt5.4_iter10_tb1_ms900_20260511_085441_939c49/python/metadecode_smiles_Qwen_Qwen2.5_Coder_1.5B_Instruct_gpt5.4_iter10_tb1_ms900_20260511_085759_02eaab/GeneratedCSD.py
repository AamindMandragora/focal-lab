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
        d_2_rollbackLimit_: int
        d_2_rollbackLimit_ = 64
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_openedGenerated_: _dafny.Seq
                        d_4_openedInside_: bool
                        d_5_openedCurrent_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_3_openedGenerated_ = out0_
                        d_4_openedInside_ = out1_
                        d_5_openedCurrent_ = out2_
                        generated = d_3_openedGenerated_
                        insideConstrainedOut = d_4_openedInside_
                        currentConstrainedOut = d_5_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_6_closedGenerated_: _dafny.Seq
                        d_7_closedInside_: bool
                        d_8_closedCurrent_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_6_closedGenerated_ = out3_
                        d_7_closedInside_ = out4_
                        d_8_closedCurrent_ = out5_
                        generated = d_6_closedGenerated_
                        insideConstrainedOut = d_7_closedInside_
                        currentConstrainedOut = d_8_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif (len(currentConstrainedOut)) >= (d_2_rollbackLimit_):
                        d_9_rolledGenerated_: _dafny.Seq
                        d_10_rolledCurrent_: _dafny.Seq
                        out6_: _dafny.Seq
                        out7_: _dafny.Seq
                        out6_, out7_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_9_rolledGenerated_ = out6_
                        d_10_rolledCurrent_ = out7_
                        generated = d_9_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_10_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_11_stablePrefix_: _dafny.Seq
                        d_11_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_12_constrainedPrompt_: _dafny.Seq
                        d_12_constrainedPrompt_ = (prompt) + (d_11_stablePrefix_)
                        d_13_remaining_: int
                        d_13_remaining_ = (maxSteps) - (d_1_steps_)
                        d_14_symbolBudget_: int
                        if ((stepTokenBudget) == (0)) or ((stepTokenBudget) > (d_13_remaining_)):
                            d_14_symbolBudget_ = d_13_remaining_
                        elif True:
                            d_14_symbolBudget_ = stepTokenBudget
                        d_15_symbolGenerated_: _dafny.Seq
                        d_16_symbolOut_: _dafny.Seq
                        d_17_hitEos_: bool
                        d_18_symbolSteps_: int
                        out8_: _dafny.Seq
                        out9_: _dafny.Seq
                        out10_: bool
                        out11_: int
                        out8_, out9_, out10_, out11_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_12_constrainedPrompt_, generated, currentConstrainedOut, d_14_symbolBudget_, eosToken)
                        d_15_symbolGenerated_ = out8_
                        d_16_symbolOut_ = out9_
                        d_17_hitEos_ = out10_
                        d_18_symbolSteps_ = out11_
                        generated = d_15_symbolGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_16_symbolOut_
                        d_1_steps_ = (d_1_steps_) + (d_18_symbolSteps_)
                        if d_17_hitEos_:
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

