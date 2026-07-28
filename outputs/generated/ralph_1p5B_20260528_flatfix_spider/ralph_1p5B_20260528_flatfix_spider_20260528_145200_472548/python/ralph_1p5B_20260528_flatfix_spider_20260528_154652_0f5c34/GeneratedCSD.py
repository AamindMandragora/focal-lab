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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer with exactly one line: SQL: <<query>>. Use the minimum number of tables needed — never add a JOIN unless the question explicitly requires data from that extra table. Use ONLY exact table and column names from the schema. No explanations, no markdown.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_2_preambleBudget_: int
            if ((maxSteps) - (d_1_steps_)) < (6):
                d_2_preambleBudget_ = (maxSteps) - (d_1_steps_)
            elif True:
                d_2_preambleBudget_ = 6
            d_3_genOut_: _dafny.Seq
            d_4_stoppedOnOpenSpan_: bool
            d_5_stoppedOnEos_: bool
            d_6_su_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_preambleBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_3_genOut_ = out0_
            d_4_stoppedOnOpenSpan_ = out1_
            d_5_stoppedOnEos_ = out2_
            d_6_su_ = out3_
            generated = d_3_genOut_
            d_1_steps_ = (d_1_steps_) + (d_6_su_)
            if d_5_stoppedOnEos_:
                cost = d_1_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            if d_4_stoppedOnOpenSpan_:
                d_7_g_: _dafny.Seq
                d_8_i_: bool
                d_9_c_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_7_g_ = out4_
                d_8_i_ = out5_
                d_9_c_ = out6_
                generated = d_7_g_
                insideConstrainedOut = d_8_i_
                currentConstrainedOut = d_9_c_
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_10_g_: _dafny.Seq
            d_11_i_: bool
            d_12_c_: _dafny.Seq
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_10_g_ = out7_
            d_11_i_ = out8_
            d_12_c_ = out9_
            generated = d_10_g_
            insideConstrainedOut = d_11_i_
            currentConstrainedOut = d_12_c_
            d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_13_g_: _dafny.Seq
                        d_14_i_: bool
                        d_15_c_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_13_g_ = out10_
                        d_14_i_ = out11_
                        d_15_c_ = out12_
                        generated = d_13_g_
                        insideConstrainedOut = d_14_i_
                        currentConstrainedOut = d_15_c_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_16_deadEnd_: bool
                        out13_: bool
                        out13_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_16_deadEnd_ = out13_
                        if d_16_deadEnd_:
                            raise _dafny.Break("0")
                        d_17_constrainedPrompt_: _dafny.Seq
                        d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_18_next_: _dafny.Seq
                        out14_: _dafny.Seq
                        out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), 40, eosToken)
                        d_18_next_ = out14_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_18_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_19_g_: _dafny.Seq
                            d_20_i_: bool
                            d_21_c_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                            d_19_g_ = out15_
                            d_20_i_ = out16_
                            d_21_c_ = out17_
                            generated = d_19_g_
                            insideConstrainedOut = d_20_i_
                            currentConstrainedOut = d_21_c_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

