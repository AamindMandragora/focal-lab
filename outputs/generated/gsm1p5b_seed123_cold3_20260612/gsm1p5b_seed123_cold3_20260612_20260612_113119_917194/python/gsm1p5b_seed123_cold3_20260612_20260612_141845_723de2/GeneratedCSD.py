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
        (d_0_helpers_).AppendTaskGuidance(lm, ((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Show each calculation inside << >> delimiters. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use ONLY: integers, variable names, and operators +, -, *, /, (, ). ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NO exponents (^, **), NO braces {}, NO LaTeX. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Example: <<n1 * p + n2>>. Final answer: <<total_expression>>."))))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_chunkSize_: int
                        d_2_chunkSize_ = (maxSteps) - (d_1_steps_)
                        if (d_2_chunkSize_) > (16):
                            d_2_chunkSize_ = 16
                        if (d_2_chunkSize_) == (0):
                            raise _dafny.Break("0")
                        d_3_chunkGenerated_: _dafny.Seq
                        d_4_stoppedOnOpenSpan_: bool
                        d_5_stoppedOnEos_: bool
                        d_6_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkSize_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_3_chunkGenerated_ = out0_
                        d_4_stoppedOnOpenSpan_ = out1_
                        d_5_stoppedOnEos_ = out2_
                        d_6_stepsUsed_ = out3_
                        d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed_)
                        generated = d_3_chunkGenerated_
                        if d_4_stoppedOnOpenSpan_:
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
                        elif d_5_stoppedOnEos_:
                            raise _dafny.Break("0")
                    elif True:
                        d_10_cg_: _dafny.Seq
                        d_11_ci_: bool
                        d_12_cc_: _dafny.Seq
                        d_13_closed_: bool
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out10_: bool
                        out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_10_cg_ = out7_
                        d_11_ci_ = out8_
                        d_12_cc_ = out9_
                        d_13_closed_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_13_closed_:
                            generated = d_10_cg_
                            insideConstrainedOut = d_11_ci_
                            currentConstrainedOut = d_12_cc_
                        elif True:
                            d_14_constrainedPrompt_: _dafny.Seq
                            d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_15_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_15_next_ = out11_
                            if (d_15_next_) == (eosToken):
                                d_16_rg_: _dafny.Seq
                                d_17_rc_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: _dafny.Seq
                                out12_, out13_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_16_rg_ = out12_
                                d_17_rc_ = out13_
                                generated = d_16_rg_
                                currentConstrainedOut = d_17_rc_
                                d_18_isComplete_: bool
                                d_18_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if (d_18_isComplete_) and ((d_1_steps_) < (maxSteps)):
                                    d_19_fg_: _dafny.Seq
                                    d_20_fi_: bool
                                    d_21_fc_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_19_fg_ = out14_
                                    d_20_fi_ = out15_
                                    d_21_fc_ = out16_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    generated = d_19_fg_
                                    insideConstrainedOut = d_20_fi_
                                    currentConstrainedOut = d_21_fc_
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                            elif True:
                                d_22_ag_: _dafny.Seq
                                d_23_ai_: bool
                                d_24_ac_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                d_22_ag_ = out17_
                                d_23_ai_ = out18_
                                d_24_ac_ = out19_
                                generated = d_22_ag_
                                insideConstrainedOut = d_23_ai_
                                currentConstrainedOut = d_24_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_25_rg_: _dafny.Seq
            d_26_rc_: _dafny.Seq
            out20_: _dafny.Seq
            out21_: _dafny.Seq
            out20_, out21_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_25_rg_ = out20_
            d_26_rc_ = out21_
            generated = d_25_rg_
            currentConstrainedOut = d_26_rc_
            d_27_isComplete_: bool
            d_27_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if d_27_isComplete_:
                d_28_fg_: _dafny.Seq
                d_29_fi_: bool
                d_30_fc_: _dafny.Seq
                out22_: _dafny.Seq
                out23_: bool
                out24_: _dafny.Seq
                out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_28_fg_ = out22_
                d_29_fi_ = out23_
                d_30_fc_ = out24_
                d_1_steps_ = (d_1_steps_) + (1)
                generated = d_28_fg_
                insideConstrainedOut = d_29_fi_
                currentConstrainedOut = d_30_fc_
            elif True:
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

