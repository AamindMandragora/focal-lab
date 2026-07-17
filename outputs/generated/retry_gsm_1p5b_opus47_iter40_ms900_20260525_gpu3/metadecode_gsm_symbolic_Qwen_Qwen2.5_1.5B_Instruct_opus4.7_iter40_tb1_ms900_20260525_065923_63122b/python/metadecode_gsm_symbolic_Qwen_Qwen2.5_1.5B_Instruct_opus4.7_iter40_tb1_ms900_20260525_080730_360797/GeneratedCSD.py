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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Read the question carefully. 'How much more is A than B' or 'how many left/remaining' means SUBTRACT (A-B). 'How many total/altogether' means ADD. Use the problem's variable names without curly braces in arithmetic. Wrap each arithmetic step as <<expression=result>>, e.g. <<total-spent=remaining>> or <<price*qty=cost>>. Use several short << >> boxes to build the answer step by step. After the final boxed expression, write a new line: '#### <final_expression>'.")))
        d_1_budget_: int
        if (maxSteps) > (400):
            d_1_budget_ = 400
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
                        d_8_next_: _dafny.Seq
                        d_9_wasConstrained_: bool
                        out4_: _dafny.Seq
                        out5_: bool
                        out4_, out5_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_8_next_ = out4_
                        d_9_wasConstrained_ = out5_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_8_next_) == (eosToken):
                            raise _dafny.Break("0")
                        d_10_ag_: _dafny.Seq
                        d_11_ai_: bool
                        d_12_ac_: _dafny.Seq
                        out6_: _dafny.Seq
                        out7_: bool
                        out8_: _dafny.Seq
                        out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_8_next_)
                        d_10_ag_ = out6_
                        d_11_ai_ = out7_
                        d_12_ac_ = out8_
                        generated = d_10_ag_
                        insideConstrainedOut = d_11_ai_
                        currentConstrainedOut = d_12_ac_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

