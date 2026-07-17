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
        (d_0_helpers_).AppendTaskGuidance(lm, (((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Show each calculation inside << >> delimiters using simple arithmetic only. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use only +, -, *, /, (, ), and numbers or variable names inside << >>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Example: She has <<n1+n2>> apples. The final answer is <<n1*n2>>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Do NOT use Python syntax, ** for exponents, or { } braces inside << >>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Open << when you want to write a calculation, close >> immediately after."))))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_chunkMax_: int
                        d_2_chunkMax_ = (maxSteps) - (d_1_steps_)
                        if (d_2_chunkMax_) > (8):
                            d_2_chunkMax_ = 8
                        d_3_chunkGenerated_: _dafny.Seq
                        d_4_stoppedOnOpenSpan_: bool
                        d_5_stoppedOnEos_: bool
                        d_6_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkMax_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_3_chunkGenerated_ = out0_
                        d_4_stoppedOnOpenSpan_ = out1_
                        d_5_stoppedOnEos_ = out2_
                        d_6_stepsUsed_ = out3_
                        d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed_)
                        generated = d_3_chunkGenerated_
                        if d_5_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_4_stoppedOnOpenSpan_:
                            d_7_og_: _dafny.Seq
                            d_8_oi_: bool
                            d_9_oc_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_7_og_ = out4_
                            d_8_oi_ = out5_
                            d_9_oc_ = out6_
                            generated = d_7_og_
                            insideConstrainedOut = d_8_oi_
                            currentConstrainedOut = d_9_oc_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_10_fg_: _dafny.Seq
                        d_11_fi_: bool
                        d_12_fc_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_10_fg_ = out7_
                        d_11_fi_ = out8_
                        d_12_fc_ = out9_
                        d_1_steps_ = (d_1_steps_) + (1)
                        generated = d_10_fg_
                        insideConstrainedOut = d_11_fi_
                        currentConstrainedOut = d_12_fc_
                    elif True:
                        d_13_constrainedPrompt_: _dafny.Seq
                        d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_14_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "{")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "}"))]), _dafny.BigRational('8e0'), eosToken)
                        d_14_next_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_14_next_) == (eosToken):
                            d_15_rg_: _dafny.Seq
                            d_16_rc_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: _dafny.Seq
                            out11_, out12_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_15_rg_ = out11_
                            d_16_rc_ = out12_
                            generated = d_15_rg_
                            currentConstrainedOut = d_16_rc_
                            d_17_isComplete_: bool
                            d_17_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if (d_17_isComplete_) and ((d_1_steps_) < (maxSteps)):
                                d_18_fg_: _dafny.Seq
                                d_19_fi_: bool
                                d_20_fc_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_18_fg_ = out13_
                                d_19_fi_ = out14_
                                d_20_fc_ = out15_
                                d_1_steps_ = (d_1_steps_) + (1)
                                generated = d_18_fg_
                                insideConstrainedOut = d_19_fi_
                                currentConstrainedOut = d_20_fc_
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            raise _dafny.Break("0")
                        elif True:
                            d_21_ag_: _dafny.Seq
                            d_22_ai_: bool
                            d_23_ac_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                            d_21_ag_ = out16_
                            d_22_ai_ = out17_
                            d_23_ac_ = out18_
                            generated = d_21_ag_
                            insideConstrainedOut = d_22_ai_
                            currentConstrainedOut = d_23_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_24_rg_: _dafny.Seq
            d_25_rc_: _dafny.Seq
            out19_: _dafny.Seq
            out20_: _dafny.Seq
            out19_, out20_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_24_rg_ = out19_
            d_25_rc_ = out20_
            generated = d_24_rg_
            currentConstrainedOut = d_25_rc_
            d_26_isComplete_: bool
            d_26_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if d_26_isComplete_:
                d_27_fg_: _dafny.Seq
                d_28_fi_: bool
                d_29_fc_: _dafny.Seq
                out21_: _dafny.Seq
                out22_: bool
                out23_: _dafny.Seq
                out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_27_fg_ = out21_
                d_28_fi_ = out22_
                d_29_fc_ = out23_
                d_1_steps_ = (d_1_steps_) + (1)
                generated = d_27_fg_
                insideConstrainedOut = d_28_fi_
                currentConstrainedOut = d_29_fc_
            elif True:
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

