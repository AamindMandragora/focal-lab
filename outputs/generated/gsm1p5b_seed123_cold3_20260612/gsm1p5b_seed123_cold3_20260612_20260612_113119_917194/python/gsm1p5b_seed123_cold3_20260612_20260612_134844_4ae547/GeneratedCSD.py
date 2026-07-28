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
        (d_0_helpers_).AppendTaskGuidance(lm, (((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Show EVERY arithmetic expression inside << >> delimiters. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use ONLY integers, variable names (like n, k, x), +, -, *, /, ( ) inside << >>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NO braces {}, NO percent %, NO ** inside << >>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "End with a << >> containing just the final numeric expression. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Example: Total cost is <<n * p>>. The answer is <<n * p>>."))))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_chunkSize_: int
        d_2_chunkSize_ = 15
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_remaining_: int
                        d_3_remaining_ = (maxSteps) - (d_1_steps_)
                        d_4_actualChunk_: int
                        if (d_3_remaining_) < (d_2_chunkSize_):
                            d_4_actualChunk_ = d_3_remaining_
                        elif True:
                            d_4_actualChunk_ = d_2_chunkSize_
                        if (d_4_actualChunk_) == (0):
                            raise _dafny.Break("0")
                        d_5_genOut_: _dafny.Seq
                        d_6_stoppedOnOpen_: bool
                        d_7_stoppedOnEos_: bool
                        d_8_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_actualChunk_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_5_genOut_ = out0_
                        d_6_stoppedOnOpen_ = out1_
                        d_7_stoppedOnEos_ = out2_
                        d_8_stepsUsed_ = out3_
                        d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                        generated = d_5_genOut_
                        if d_7_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif d_6_stoppedOnOpen_:
                            d_9_og_: _dafny.Seq
                            d_10_oi_: bool
                            d_11_oc_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_9_og_ = out4_
                            d_10_oi_ = out5_
                            d_11_oc_ = out6_
                            generated = d_9_og_
                            insideConstrainedOut = d_10_oi_
                            currentConstrainedOut = d_11_oc_
                    elif True:
                        d_12_cg_: _dafny.Seq
                        d_13_ci_: bool
                        d_14_cc_: _dafny.Seq
                        d_15_closed_: bool
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out10_: bool
                        out7_, out8_, out9_, out10_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_12_cg_ = out7_
                        d_13_ci_ = out8_
                        d_14_cc_ = out9_
                        d_15_closed_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_15_closed_:
                            generated = d_12_cg_
                            insideConstrainedOut = d_13_ci_
                            currentConstrainedOut = d_14_cc_
                        elif True:
                            d_16_constrainedPrompt_: _dafny.Seq
                            d_16_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_17_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_16_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_17_next_ = out11_
                            if (d_17_next_) == (eosToken):
                                d_18_rg_: _dafny.Seq
                                d_19_rc_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: _dafny.Seq
                                out12_, out13_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_18_rg_ = out12_
                                d_19_rc_ = out13_
                                generated = d_18_rg_
                                currentConstrainedOut = d_19_rc_
                                d_20_isComplete_: bool
                                d_20_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if (d_20_isComplete_) and ((d_1_steps_) < (maxSteps)):
                                    d_21_fg_: _dafny.Seq
                                    d_22_fi_: bool
                                    d_23_fc_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_21_fg_ = out14_
                                    d_22_fi_ = out15_
                                    d_23_fc_ = out16_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    generated = d_21_fg_
                                    insideConstrainedOut = d_22_fi_
                                    currentConstrainedOut = d_23_fc_
                                elif True:
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                            elif True:
                                d_24_ag_: _dafny.Seq
                                d_25_ai_: bool
                                d_26_ac_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_17_next_)
                                d_24_ag_ = out17_
                                d_25_ai_ = out18_
                                d_26_ac_ = out19_
                                generated = d_24_ag_
                                insideConstrainedOut = d_25_ai_
                                currentConstrainedOut = d_26_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_27_rg_: _dafny.Seq
            d_28_rc_: _dafny.Seq
            out20_: _dafny.Seq
            out21_: _dafny.Seq
            out20_, out21_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_27_rg_ = out20_
            d_28_rc_ = out21_
            generated = d_27_rg_
            currentConstrainedOut = d_28_rc_
            d_29_isComplete_: bool
            d_29_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if d_29_isComplete_:
                d_30_fg_: _dafny.Seq
                d_31_fi_: bool
                d_32_fc_: _dafny.Seq
                out22_: _dafny.Seq
                out23_: bool
                out24_: _dafny.Seq
                out22_, out23_, out24_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_30_fg_ = out22_
                d_31_fi_ = out23_
                d_32_fc_ = out24_
                d_1_steps_ = (d_1_steps_) + (1)
                generated = d_30_fg_
                insideConstrainedOut = d_31_fi_
                currentConstrainedOut = d_32_fc_
            elif True:
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

