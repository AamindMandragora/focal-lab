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
        (d_0_helpers_).AppendTaskGuidance(lm, (((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Show each calculation inside << >> delimiters. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use only +, -, *, /, (, ), numbers, and simple variable names inside << >>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Do NOT use { or } or ** inside << >>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Example: She earns <<n1 * n2>> dollars. Final answer: <<n1 * n2>>. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Keep each << >> expression short and close >> immediately after the expression."))))
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
                        elif (d_2_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                            d_3_og_: _dafny.Seq
                            d_4_oi_: bool
                            d_5_oc_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_3_og_ = out1_
                            d_4_oi_ = out2_
                            d_5_oc_ = out3_
                            generated = d_3_og_
                            insideConstrainedOut = d_4_oi_
                            currentConstrainedOut = d_5_oc_
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_6_fg_: _dafny.Seq
                        d_7_fi_: bool
                        d_8_fc_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_6_fg_ = out4_
                        d_7_fi_ = out5_
                        d_8_fc_ = out6_
                        d_1_steps_ = (d_1_steps_) + (1)
                        generated = d_6_fg_
                        insideConstrainedOut = d_7_fi_
                        currentConstrainedOut = d_8_fc_
                    elif True:
                        d_9_constrainedPrompt_: _dafny.Seq
                        d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_10_next_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 20, eosToken)
                        d_10_next_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_10_next_) == (eosToken):
                            d_11_rg_: _dafny.Seq
                            d_12_rc_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: _dafny.Seq
                            out8_, out9_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_11_rg_ = out8_
                            d_12_rc_ = out9_
                            generated = d_11_rg_
                            currentConstrainedOut = d_12_rc_
                            d_13_isComplete_: bool
                            d_13_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if (d_13_isComplete_) and ((d_1_steps_) < (maxSteps)):
                                d_14_fg_: _dafny.Seq
                                d_15_fi_: bool
                                d_16_fc_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_14_fg_ = out10_
                                d_15_fi_ = out11_
                                d_16_fc_ = out12_
                                d_1_steps_ = (d_1_steps_) + (1)
                                generated = d_14_fg_
                                insideConstrainedOut = d_15_fi_
                                currentConstrainedOut = d_16_fc_
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            raise _dafny.Break("0")
                        elif True:
                            d_17_ag_: _dafny.Seq
                            d_18_ai_: bool
                            d_19_ac_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                            d_17_ag_ = out13_
                            d_18_ai_ = out14_
                            d_19_ac_ = out15_
                            generated = d_17_ag_
                            insideConstrainedOut = d_18_ai_
                            currentConstrainedOut = d_19_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_20_rg_: _dafny.Seq
            d_21_rc_: _dafny.Seq
            out16_: _dafny.Seq
            out17_: _dafny.Seq
            out16_, out17_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_20_rg_ = out16_
            d_21_rc_ = out17_
            generated = d_20_rg_
            currentConstrainedOut = d_21_rc_
            d_22_isComplete_: bool
            d_22_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
            if d_22_isComplete_:
                d_23_fg_: _dafny.Seq
                d_24_fi_: bool
                d_25_fc_: _dafny.Seq
                out18_: _dafny.Seq
                out19_: bool
                out20_: _dafny.Seq
                out18_, out19_, out20_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_23_fg_ = out18_
                d_24_fi_ = out19_
                d_25_fc_ = out20_
                d_1_steps_ = (d_1_steps_) + (1)
                generated = d_23_fg_
                insideConstrainedOut = d_24_fi_
                currentConstrainedOut = d_25_fc_
            elif True:
                insideConstrainedOut = False
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

