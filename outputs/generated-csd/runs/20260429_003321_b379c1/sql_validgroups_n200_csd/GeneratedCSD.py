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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_narrowThreshold_: int
        d_1_narrowThreshold_ = 12
        d_2_steps_: int
        d_2_steps_ = 0
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_3_next_ = out0_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                            if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_4_complete_: bool
                        d_4_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_4_complete_:
                            d_5_closedGenerated_: _dafny.Seq
                            d_6_closedInside_: bool
                            d_7_closedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_5_closedGenerated_ = out1_
                            d_6_closedInside_ = out2_
                            d_7_closedCurrent_ = out3_
                            generated = d_5_closedGenerated_
                            insideConstrainedOut = d_6_closedInside_
                            currentConstrainedOut = d_7_closedCurrent_
                            d_2_steps_ = (d_2_steps_) + (1)
                        elif True:
                            d_8_constrainedPrompt_: _dafny.Seq
                            d_8_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_9_validCount_: int
                            out4_: int
                            out4_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                            d_9_validCount_ = out4_
                            d_10_useWide_: bool
                            d_10_useWide_ = False
                            if (d_9_validCount_) > (d_1_narrowThreshold_):
                                d_10_useWide_ = True
                                if (len(validTokenGroups)) > (0):
                                    d_11_flatPreferred_: _dafny.Seq
                                    out5_: _dafny.Seq
                                    out5_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
                                    d_11_flatPreferred_ = out5_
                                    if (len(d_11_flatPreferred_)) > (0):
                                        d_12_anyPreferredValid_: bool
                                        out6_: bool
                                        out6_ = (d_0_helpers_).GroupHasValidMember(parser, currentConstrainedOut, d_11_flatPreferred_)
                                        d_12_anyPreferredValid_ = out6_
                                        if not(d_12_anyPreferredValid_):
                                            d_10_useWide_ = False
                            if ((not(d_10_useWide_)) or ((stepTokenBudget) == (0))) or (((maxSteps) - (d_2_steps_)) < (stepTokenBudget)):
                                d_13_next_: _dafny.Seq
                                out7_: _dafny.Seq
                                out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_13_next_ = out7_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if (d_13_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_14_appendedGenerated_: _dafny.Seq
                                    d_15_appendedInside_: bool
                                    d_16_appendedCurrent_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out9_: bool
                                    out10_: _dafny.Seq
                                    out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                                    d_14_appendedGenerated_ = out8_
                                    d_15_appendedInside_ = out9_
                                    d_16_appendedCurrent_ = out10_
                                    generated = d_14_appendedGenerated_
                                    insideConstrainedOut = d_15_appendedInside_
                                    currentConstrainedOut = d_16_appendedCurrent_
                            elif True:
                                d_17_stablePrefix_: _dafny.Seq
                                d_17_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_18_symbolOut_: _dafny.Seq
                                d_19_hitEos_: bool
                                d_20_stepsUsed_: int
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: int
                                out11_, out12_, out13_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, stepTokenBudget, eosToken)
                                d_18_symbolOut_ = out11_
                                d_19_hitEos_ = out12_
                                d_20_stepsUsed_ = out13_
                                generated = (d_17_stablePrefix_) + (d_18_symbolOut_)
                                currentConstrainedOut = d_18_symbolOut_
                                d_2_steps_ = (d_2_steps_) + (d_20_stepsUsed_)
                                if d_19_hitEos_:
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_2_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

