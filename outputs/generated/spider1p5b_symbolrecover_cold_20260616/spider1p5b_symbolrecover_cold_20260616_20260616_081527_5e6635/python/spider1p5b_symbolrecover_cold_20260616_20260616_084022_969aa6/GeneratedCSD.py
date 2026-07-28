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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly: SQL: <<query>> where query is a single valid SQL SELECT statement using only the schema tables and columns provided. No subquery nesting beyond what is needed. No markdown. No explanation.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_constrainedTokens_: int
        d_2_constrainedTokens_ = 0
        d_3_maxConstrainedTokens_: int
        d_3_maxConstrainedTokens_ = 120
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_constrainedTokens_ = 0
                    elif True:
                        d_5_cg_: _dafny.Seq
                        d_6_ci_: bool
                        d_7_cc_: _dafny.Seq
                        d_8_closed_: bool
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out4_: bool
                        out1_, out2_, out3_, out4_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_5_cg_ = out1_
                        d_6_ci_ = out2_
                        d_7_cc_ = out3_
                        d_8_closed_ = out4_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_8_closed_:
                            generated = d_5_cg_
                            insideConstrainedOut = d_6_ci_
                            currentConstrainedOut = d_7_cc_
                            raise _dafny.Break("0")
                        elif (d_1_steps_) >= (maxSteps):
                            raise _dafny.Break("0")
                        elif (d_2_constrainedTokens_) >= (d_3_maxConstrainedTokens_):
                            d_9_rg_: _dafny.Seq
                            d_10_rc_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: _dafny.Seq
                            out5_, out6_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_9_rg_ = out5_
                            d_10_rc_ = out6_
                            generated = d_9_rg_
                            currentConstrainedOut = d_10_rc_
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_11_fcg_: _dafny.Seq
                                d_12_fci_: bool
                                d_13_fcc_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_11_fcg_ = out7_
                                d_12_fci_ = out8_
                                d_13_fcc_ = out9_
                                generated = d_11_fcg_
                                insideConstrainedOut = d_12_fci_
                                currentConstrainedOut = d_13_fcc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_14_constrainedPrompt_: _dafny.Seq
                            d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_15_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            if (_dafny.euclidian_modulus(d_2_constrainedTokens_, 4)) == (3):
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('3e0'), eosToken)
                                d_15_next_ = out10_
                            elif True:
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                d_15_next_ = out11_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_15_next_) == (eosToken):
                                if (parser).IsCompletePrefix(currentConstrainedOut):
                                    d_16_fcg_: _dafny.Seq
                                    d_17_fci_: bool
                                    d_18_fcc_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_16_fcg_ = out12_
                                    d_17_fci_ = out13_
                                    d_18_fcc_ = out14_
                                    generated = d_16_fcg_
                                    insideConstrainedOut = d_17_fci_
                                    currentConstrainedOut = d_18_fcc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                d_19_ag_: _dafny.Seq
                                d_20_ai_: bool
                                d_21_ac_: _dafny.Seq
                                out15_: _dafny.Seq
                                out16_: bool
                                out17_: _dafny.Seq
                                out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                d_19_ag_ = out15_
                                d_20_ai_ = out16_
                                d_21_ac_ = out17_
                                generated = d_19_ag_
                                insideConstrainedOut = d_20_ai_
                                currentConstrainedOut = d_21_ac_
                                d_2_constrainedTokens_ = (d_2_constrainedTokens_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

