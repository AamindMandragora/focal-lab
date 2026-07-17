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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer the question using ONLY the tables and columns in the provided database schema. Output exactly: SQL: <<your SQL query here>> with no other text. Use only single-quoted strings, no semicolons, no backticks.")))
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
                            elif (((d_2_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<")))) and ((len(generated)) >= (2))) and (((generated)[(len(generated)) - (2)]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<")))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_3_cg_: _dafny.Seq
                        d_4_ci_: bool
                        d_5_cc_: _dafny.Seq
                        d_6_closed_: bool
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out4_: bool
                        out1_, out2_, out3_, out4_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_3_cg_ = out1_
                        d_4_ci_ = out2_
                        d_5_cc_ = out3_
                        d_6_closed_ = out4_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_6_closed_:
                            generated = d_3_cg_
                            insideConstrainedOut = d_4_ci_
                            currentConstrainedOut = d_5_cc_
                            raise _dafny.Break("0")
                        elif True:
                            if (d_1_steps_) < (maxSteps):
                                d_7_constrainedPrompt_: _dafny.Seq
                                d_7_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_8_next_: _dafny.Seq
                                out5_: _dafny.Seq
                                out5_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_8_next_ = out5_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_8_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_9_valid_: bool
                                    out6_: bool
                                    out6_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_8_next_)
                                    d_9_valid_ = out6_
                                    if d_9_valid_:
                                        d_10_ag_: _dafny.Seq
                                        d_11_ai_: bool
                                        d_12_ac_: _dafny.Seq
                                        out7_: _dafny.Seq
                                        out8_: bool
                                        out9_: _dafny.Seq
                                        out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_8_next_)
                                        d_10_ag_ = out7_
                                        d_11_ai_ = out8_
                                        d_12_ac_ = out9_
                                        generated = d_10_ag_
                                        insideConstrainedOut = d_11_ai_
                                        currentConstrainedOut = d_12_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_13_closeBudget_: int
            d_13_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_14_cg_: _dafny.Seq
            d_15_ci_: bool
            d_16_cc_: _dafny.Seq
            out10_: _dafny.Seq
            out11_: bool
            out12_: _dafny.Seq
            out10_, out11_, out12_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_13_closeBudget_)
            d_14_cg_ = out10_
            d_15_ci_ = out11_
            d_16_cc_ = out12_
            generated = d_14_cg_
            insideConstrainedOut = d_15_ci_
            currentConstrainedOut = d_16_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

