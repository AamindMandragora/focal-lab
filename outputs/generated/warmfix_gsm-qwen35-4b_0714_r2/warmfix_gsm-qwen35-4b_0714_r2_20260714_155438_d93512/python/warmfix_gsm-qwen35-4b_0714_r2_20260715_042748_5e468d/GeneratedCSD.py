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
        out0_: _dafny.Seq
        out1_: bool
        out2_: _dafny.Seq
        out3_: int
        out0_, out1_, out2_, out3_ = default__.AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken)
        generated = out0_
        insideConstrainedOut = out1_
        currentConstrainedOut = out2_
        cost = out3_
        if ((maxSteps) > (0)) and ((cost) <= (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

    @staticmethod
    def AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
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
        if (maxSteps) == (0):
            cost = 0
        elif True:
            d_1_guidance_: _dafny.Seq
            d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. At the very end, write your final arithmetic answer as a single expression inside << >> using only variable names, numbers, +, -, *, /, //, %, (, ), int(). No LaTeX, no {}, no **. Write exactly one <<expression>> at the end."))
            (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
            d_2_steps_: int
            d_2_steps_ = 0
            with _dafny.label("1_0"):
                while (d_2_steps_) < (maxSteps):
                    with _dafny.c_label("1_0"):
                        if not(insideConstrainedOut):
                            d_3_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_3_next_ = out0_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_3_next_) == (eosToken):
                                raise _dafny.Break("1_0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                                if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif True:
                            d_4_cg_: _dafny.Seq
                            d_5_ci_: bool
                            d_6_cc_: _dafny.Seq
                            d_7_closed_: bool
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out4_: bool
                            out1_, out2_, out3_, out4_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_4_cg_ = out1_
                            d_5_ci_ = out2_
                            d_6_cc_ = out3_
                            d_7_closed_ = out4_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if d_7_closed_:
                                generated = d_4_cg_
                                insideConstrainedOut = d_5_ci_
                                currentConstrainedOut = d_6_cc_
                            elif True:
                                if (d_2_steps_) < (maxSteps):
                                    d_8_constrainedPrompt_: _dafny.Seq
                                    d_8_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                    d_9_next_: _dafny.Seq
                                    out5_: _dafny.Seq
                                    out5_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, eosToken)
                                    d_9_next_ = out5_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    if (d_9_next_) == (eosToken):
                                        raise _dafny.Break("1_0")
                                    elif True:
                                        d_10_valid_: bool
                                        out6_: bool
                                        out6_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_9_next_)
                                        d_10_valid_ = out6_
                                        if d_10_valid_:
                                            d_11_ag_: _dafny.Seq
                                            d_12_ai_: bool
                                            d_13_ac_: _dafny.Seq
                                            out7_: _dafny.Seq
                                            out8_: bool
                                            out9_: _dafny.Seq
                                            out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                                            d_11_ag_ = out7_
                                            d_12_ai_ = out8_
                                            d_13_ac_ = out9_
                                            generated = d_11_ag_
                                            insideConstrainedOut = d_12_ai_
                                            currentConstrainedOut = d_13_ac_
                        pass
                pass
            cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

