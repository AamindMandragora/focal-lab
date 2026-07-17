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
                        d_6_stablePrefix_: _dafny.Seq
                        d_6_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_7_constrainedPrompt_: _dafny.Seq
                        d_7_constrainedPrompt_ = (prompt) + (d_6_stablePrefix_)
                        d_8_doSpeculate_: bool
                        d_8_doSpeculate_ = ((len(currentConstrainedOut)) >= (4)) and (((maxSteps) - (d_1_steps_)) >= (8))
                        d_9_didRollback_: bool
                        d_9_didRollback_ = False
                        if d_8_doSpeculate_:
                            d_10_specBudget_: int
                            d_10_specBudget_ = 4
                            if (d_10_specBudget_) > ((maxSteps) - (d_1_steps_)):
                                d_10_specBudget_ = (maxSteps) - (d_1_steps_)
                            d_11_candTok_: _dafny.Seq
                            d_12_candPre_: _dafny.Seq
                            d_13_hitComplete_: bool
                            d_14_hitEos_: bool
                            d_15_specStepsUsed_: int
                            out4_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: bool
                            out8_: int
                            out4_, out5_, out6_, out7_, out8_ = (d_0_helpers_).SpeculativeConstrainedRollout(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, d_10_specBudget_, eosToken)
                            d_11_candTok_ = out4_
                            d_12_candPre_ = out5_
                            d_13_hitComplete_ = out6_
                            d_14_hitEos_ = out7_
                            d_15_specStepsUsed_ = out8_
                            d_1_steps_ = (d_1_steps_) + (d_15_specStepsUsed_)
                            if ((((not(d_13_hitComplete_)) and (not(d_14_hitEos_))) and ((d_15_specStepsUsed_) < (d_10_specBudget_))) and ((len(currentConstrainedOut)) >= (10))) and ((d_1_steps_) < (maxSteps)):
                                d_16_rolledG_: _dafny.Seq
                                d_17_rolledC_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: _dafny.Seq
                                out9_, out10_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                                d_16_rolledG_ = out9_
                                d_17_rolledC_ = out10_
                                generated = d_16_rolledG_
                                currentConstrainedOut = d_17_rolledC_
                                insideConstrainedOut = True
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_9_didRollback_ = True
                        if (not(d_9_didRollback_)) and ((d_1_steps_) < (maxSteps)):
                            d_18_stablePrefix2_: _dafny.Seq
                            d_18_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_19_constrainedPrompt2_: _dafny.Seq
                            d_19_constrainedPrompt2_ = (prompt) + (d_18_stablePrefix2_)
                            d_20_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            if (len(currentConstrainedOut)) < (6):
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_19_constrainedPrompt2_, currentConstrainedOut, _dafny.BigRational('7e-1'), eosToken)
                                d_20_next_ = out11_
                            elif True:
                                out12_: _dafny.Seq
                                out12_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_19_constrainedPrompt2_, currentConstrainedOut, eosToken)
                                d_20_next_ = out12_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_20_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_21_appendedGenerated_: _dafny.Seq
                                d_22_appendedInside_: bool
                                d_23_appendedCurrent_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                                d_21_appendedGenerated_ = out13_
                                d_22_appendedInside_ = out14_
                                d_23_appendedCurrent_ = out15_
                                generated = d_21_appendedGenerated_
                                insideConstrainedOut = d_22_appendedInside_
                                currentConstrainedOut = d_23_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

