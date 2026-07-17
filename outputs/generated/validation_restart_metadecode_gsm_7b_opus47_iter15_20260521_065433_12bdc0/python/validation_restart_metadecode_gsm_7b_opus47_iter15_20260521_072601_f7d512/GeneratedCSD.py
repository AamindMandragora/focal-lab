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
                        d_3_closedG_: _dafny.Seq
                        d_4_closedInside_: bool
                        d_5_closedCur_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_3_closedG_ = out1_
                        d_4_closedInside_ = out2_
                        d_5_closedCur_ = out3_
                        generated = d_3_closedG_
                        insideConstrainedOut = d_4_closedInside_
                        currentConstrainedOut = d_5_closedCur_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_6_stablePrefix_: _dafny.Seq
                        d_6_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_7_constrainedPrompt_: _dafny.Seq
                        d_7_constrainedPrompt_ = (prompt) + (d_6_stablePrefix_)
                        d_8_candidates_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, 3, eosToken)
                        d_8_candidates_ = out4_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_9_picked_: _dafny.Seq
                        d_9_picked_ = eosToken
                        d_10_i_: int
                        d_10_i_ = 0
                        with _dafny.label("0_1_1_0"):
                            while (d_10_i_) < (len(d_8_candidates_)):
                                with _dafny.c_label("0_1_1_0"):
                                    if ((d_8_candidates_)[d_10_i_]) != (eosToken):
                                        d_9_picked_ = (d_8_candidates_)[d_10_i_]
                                        raise _dafny.Break("0_1_1_0")
                                    d_10_i_ = (d_10_i_) + (1)
                                    pass
                            pass
                        if (d_9_picked_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_11_appG_: _dafny.Seq
                            d_12_appInside_: bool
                            d_13_appCur_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_picked_)
                            d_11_appG_ = out5_
                            d_12_appInside_ = out6_
                            d_13_appCur_ = out7_
                            generated = d_11_appG_
                            insideConstrainedOut = d_12_appInside_
                            currentConstrainedOut = d_13_appCur_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

