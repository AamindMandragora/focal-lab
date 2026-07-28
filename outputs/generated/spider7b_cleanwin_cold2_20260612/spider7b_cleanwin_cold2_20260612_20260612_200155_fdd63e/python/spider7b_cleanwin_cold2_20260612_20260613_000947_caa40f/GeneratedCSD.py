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
        d_2_guidance_: _dafny.Seq
        d_2_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output a single valid SQL query. Format exactly: SQL: <<your SQL query>>. Use only the table and column names from the provided schema. No explanation, no markdown."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_2_guidance_)
        d_3_closeReserve_: int
        d_3_closeReserve_ = 2
        d_4_stepParity_: int
        d_4_stepParity_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_5_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_5_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                            if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_4_stepParity_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_6_closedGenerated_: _dafny.Seq
                        d_7_closedInside_: bool
                        d_8_closedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_6_closedGenerated_ = out1_
                        d_7_closedInside_ = out2_
                        d_8_closedCurrent_ = out3_
                        generated = d_6_closedGenerated_
                        insideConstrainedOut = d_7_closedInside_
                        currentConstrainedOut = d_8_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif ((d_1_steps_) + (d_3_closeReserve_)) >= (maxSteps):
                        d_9_rolledGenerated_: _dafny.Seq
                        d_10_rolledCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: _dafny.Seq
                        out4_, out5_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_9_rolledGenerated_ = out4_
                        d_10_rolledCurrent_ = out5_
                        generated = d_9_rolledGenerated_
                        currentConstrainedOut = d_10_rolledCurrent_
                        insideConstrainedOut = True
                        d_11_cg_: _dafny.Seq
                        d_12_ci_: bool
                        d_13_cc_: _dafny.Seq
                        d_14_closed_: bool
                        out6_: _dafny.Seq
                        out7_: bool
                        out8_: _dafny.Seq
                        out9_: bool
                        out6_, out7_, out8_, out9_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_11_cg_ = out6_
                        d_12_ci_ = out7_
                        d_13_cc_ = out8_
                        d_14_closed_ = out9_
                        d_1_steps_ = (d_1_steps_) + (1)
                        generated = d_11_cg_
                        insideConstrainedOut = d_12_ci_
                        currentConstrainedOut = d_13_cc_
                        raise _dafny.Break("0")
                    elif True:
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_16_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if (_dafny.euclidian_modulus(d_4_stepParity_, 2)) == (0):
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), 20, eosToken)
                            d_16_next_ = out10_
                        elif True:
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                            d_16_next_ = out11_
                        d_4_stepParity_ = (d_4_stepParity_) + (1)
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_16_next_) == (eosToken):
                            d_17_rolledGenerated_: _dafny.Seq
                            d_18_rolledCurrent_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: _dafny.Seq
                            out12_, out13_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_17_rolledGenerated_ = out12_
                            d_18_rolledCurrent_ = out13_
                            generated = d_17_rolledGenerated_
                            currentConstrainedOut = d_18_rolledCurrent_
                            insideConstrainedOut = True
                            if (d_1_steps_) < (maxSteps):
                                d_19_cg_: _dafny.Seq
                                d_20_ci_: bool
                                d_21_cc_: _dafny.Seq
                                d_22_closed_: bool
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out17_: bool
                                out14_, out15_, out16_, out17_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                                d_19_cg_ = out14_
                                d_20_ci_ = out15_
                                d_21_cc_ = out16_
                                d_22_closed_ = out17_
                                d_1_steps_ = (d_1_steps_) + (1)
                                generated = d_19_cg_
                                insideConstrainedOut = d_20_ci_
                                currentConstrainedOut = d_21_cc_
                            raise _dafny.Break("0")
                        elif True:
                            d_23_appendedGenerated_: _dafny.Seq
                            d_24_appendedInside_: bool
                            d_25_appendedCurrent_: _dafny.Seq
                            out18_: _dafny.Seq
                            out19_: bool
                            out20_: _dafny.Seq
                            out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                            d_23_appendedGenerated_ = out18_
                            d_24_appendedInside_ = out19_
                            d_25_appendedCurrent_ = out20_
                            generated = d_23_appendedGenerated_
                            insideConstrainedOut = d_24_appendedInside_
                            currentConstrainedOut = d_25_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

