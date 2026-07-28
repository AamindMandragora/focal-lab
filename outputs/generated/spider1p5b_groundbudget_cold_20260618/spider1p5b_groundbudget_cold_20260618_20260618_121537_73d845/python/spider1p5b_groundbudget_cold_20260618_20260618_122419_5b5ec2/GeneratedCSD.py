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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You must output exactly: SQL: <<QUERY>> where QUERY is a valid SQL query using only the schema provided. The << and >> delimiters are required. Put the complete SQL query between << and >>. Example: SQL: <<SELECT col FROM table>>"))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        with _dafny.label("0"):
            while ((d_2_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_3_chunkBudget_: int
                    d_3_chunkBudget_ = (maxSteps) - (d_2_steps_)
                    if (d_3_chunkBudget_) > (20):
                        d_3_chunkBudget_ = 20
                    d_4_generatedOut_: _dafny.Seq
                    d_5_stoppedOnOpenSpan_: bool
                    d_6_stoppedOnEos_: bool
                    d_7_stepsUsed_: int
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: bool
                    out3_: int
                    out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                    d_4_generatedOut_ = out0_
                    d_5_stoppedOnOpenSpan_ = out1_
                    d_6_stoppedOnEos_ = out2_
                    d_7_stepsUsed_ = out3_
                    d_2_steps_ = (d_2_steps_) + (d_7_stepsUsed_)
                    generated = d_4_generatedOut_
                    if d_5_stoppedOnOpenSpan_:
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif d_6_stoppedOnEos_:
                        raise _dafny.Break("0")
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_8_closeBudget_: int
            d_8_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_9_cg_: _dafny.Seq
            d_10_ci_: bool
            d_11_cc_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_8_closeBudget_)
            d_9_cg_ = out4_
            d_10_ci_ = out5_
            d_11_cc_ = out6_
            generated = d_9_cg_
            insideConstrainedOut = d_10_ci_
            currentConstrainedOut = d_11_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

