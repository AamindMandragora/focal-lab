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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve briefly step by step. Wrap every arithmetic step in << >>, for example <<3+4=7>> or <<6*5=30>>. Each boxed expression must contain '=' and a numeric result, then close with >>. Keep expressions short. After the final boxed expression, write '#### <number>' on a new line.")))
        d_1_budget_: int
        if (maxSteps) > (150):
            d_1_budget_ = 150
        elif True:
            d_1_budget_ = maxSteps
        d_2_steps_: int
        d_2_steps_ = 0
        with _dafny.label("0"):
            while (d_2_steps_) < (d_1_budget_):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_3_next_ = out0_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("0")
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                        if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_4_closedG_: _dafny.Seq
                        d_5_closedI_: bool
                        d_6_closedC_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_4_closedG_ = out1_
                        d_5_closedI_ = out2_
                        d_6_closedC_ = out3_
                        generated = d_4_closedG_
                        insideConstrainedOut = d_5_closedI_
                        currentConstrainedOut = d_6_closedC_
                        d_2_steps_ = (d_2_steps_) + (1)
                    elif True:
                        d_7_constrainedPrompt_: _dafny.Seq
                        d_7_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_8_eqCount_: int
                        out4_: int
                        out4_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                        d_8_eqCount_ = out4_
                        d_9_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if ((d_8_eqCount_) == (0)) and ((len(currentConstrainedOut)) >= (3)):
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))]), _dafny.BigRational('8e0'), eosToken)
                            d_9_next_ = out5_
                        elif True:
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('3e0'), eosToken)
                            d_9_next_ = out6_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_9_next_) == (eosToken):
                            raise _dafny.Break("0")
                        d_10_ag_: _dafny.Seq
                        d_11_ai_: bool
                        d_12_ac_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                        d_10_ag_ = out7_
                        d_11_ai_ = out8_
                        d_12_ac_ = out9_
                        generated = d_10_ag_
                        insideConstrainedOut = d_11_ai_
                        currentConstrainedOut = d_12_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

