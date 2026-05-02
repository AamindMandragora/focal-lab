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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, eosToken):
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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 6
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_rem0_: int
                        d_3_rem0_ = (maxSteps) - (d_1_steps_)
                        if (d_3_rem0_) == (0):
                            raise _dafny.Break("0")
                        elif True:
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
                            d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_7_rem1_: int
                        d_7_rem1_ = (maxSteps) - (d_1_steps_)
                        if (d_7_rem1_) == (0):
                            raise _dafny.Break("0")
                        elif True:
                            d_8_complete_: bool
                            d_8_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            d_9_validCount_: int
                            out3_: int
                            out3_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_9_validCount_ = out3_
                            d_10_stablePrefix_: _dafny.Seq
                            d_10_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_11_constrainedPrompt_: _dafny.Seq
                            d_11_constrainedPrompt_ = (prompt) + (d_10_stablePrefix_)
                            if (d_8_complete_) and (((d_9_validCount_) == (0)) or ((d_7_rem1_) == (1))):
                                d_12_closedGenerated_: _dafny.Seq
                                d_13_closedInside_: bool
                                d_14_closedCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_12_closedGenerated_ = out4_
                                d_13_closedInside_ = out5_
                                d_14_closedCurrent_ = out6_
                                generated = d_12_closedGenerated_
                                insideConstrainedOut = d_13_closedInside_
                                currentConstrainedOut = d_14_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif (d_8_complete_) or ((d_9_validCount_) <= (d_2_narrowThreshold_)):
                                d_15_next_: _dafny.Seq
                                out7_: _dafny.Seq
                                out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_15_next_ = out7_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if d_8_complete_:
                                    if (d_15_next_) == (eosToken):
                                        raise _dafny.Break("0")
                                elif True:
                                    if (d_15_next_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_16_appendedGenerated_: _dafny.Seq
                                        d_17_appendedInside_: bool
                                        d_18_appendedCurrent_: _dafny.Seq
                                        out8_: _dafny.Seq
                                        out9_: bool
                                        out10_: _dafny.Seq
                                        out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                        d_16_appendedGenerated_ = out8_
                                        d_17_appendedInside_ = out9_
                                        d_18_appendedCurrent_ = out10_
                                        generated = d_16_appendedGenerated_
                                        insideConstrainedOut = d_17_appendedInside_
                                        currentConstrainedOut = d_18_appendedCurrent_
                            elif True:
                                d_19_symbolOut_: _dafny.Seq
                                d_20_hitEos_: bool
                                d_21_stepsUsed_: int
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: int
                                out11_, out12_, out13_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, d_7_rem1_, eosToken)
                                d_19_symbolOut_ = out11_
                                d_20_hitEos_ = out12_
                                d_21_stepsUsed_ = out13_
                                generated = (d_10_stablePrefix_) + (d_19_symbolOut_)
                                insideConstrainedOut = True
                                currentConstrainedOut = d_19_symbolOut_
                                d_1_steps_ = (d_1_steps_) + (d_21_stepsUsed_)
                                if d_20_hitEos_:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

