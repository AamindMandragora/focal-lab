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
        d_2_lookaheadDone_: bool
        d_2_lookaheadDone_ = False
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
                                d_2_lookaheadDone_ = False
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_4_closedG_: _dafny.Seq
                        d_5_closedInside_: bool
                        d_6_closedCur_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_4_closedG_ = out1_
                        d_5_closedInside_ = out2_
                        d_6_closedCur_ = out3_
                        generated = d_4_closedG_
                        insideConstrainedOut = d_5_closedInside_
                        currentConstrainedOut = d_6_closedCur_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_lookaheadDone_ = False
                    elif True:
                        d_7_stablePrefix_: _dafny.Seq
                        d_7_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_8_constrainedPrompt_: _dafny.Seq
                        d_8_constrainedPrompt_ = (prompt) + (d_7_stablePrefix_)
                        if (not(d_2_lookaheadDone_)) and (((maxSteps) - (d_1_steps_)) >= (5)):
                            d_9_lookBudget_: int
                            d_9_lookBudget_ = 4
                            d_10_specTok_: _dafny.Seq
                            d_11_specPre_: _dafny.Seq
                            d_12_specComplete_: bool
                            d_13_specEos_: bool
                            d_14_specSteps_: int
                            out4_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: bool
                            out8_: int
                            out4_, out5_, out6_, out7_, out8_ = (d_0_helpers_).SpeculativeConstrainedRollout(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, d_9_lookBudget_, eosToken)
                            d_10_specTok_ = out4_
                            d_11_specPre_ = out5_
                            d_12_specComplete_ = out6_
                            d_13_specEos_ = out7_
                            d_14_specSteps_ = out8_
                            d_15_charged_: int
                            if (d_14_specSteps_) >= (1):
                                d_15_charged_ = d_14_specSteps_
                            elif True:
                                d_15_charged_ = 1
                            if ((d_1_steps_) + (d_15_charged_)) > (maxSteps):
                                d_15_charged_ = (maxSteps) - (d_1_steps_)
                                if (d_15_charged_) == (0):
                                    d_15_charged_ = 1
                            if ((d_1_steps_) + (d_15_charged_)) > (maxSteps):
                                d_1_steps_ = maxSteps
                            elif True:
                                d_1_steps_ = (d_1_steps_) + (d_15_charged_)
                            d_2_lookaheadDone_ = True
                        elif True:
                            d_16_next_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_16_next_ = out9_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_16_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_17_appG_: _dafny.Seq
                                d_18_appInside_: bool
                                d_19_appCur_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                                d_17_appG_ = out10_
                                d_18_appInside_ = out11_
                                d_19_appCur_ = out12_
                                generated = d_17_appG_
                                insideConstrainedOut = d_18_appInside_
                                currentConstrainedOut = d_19_appCur_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

