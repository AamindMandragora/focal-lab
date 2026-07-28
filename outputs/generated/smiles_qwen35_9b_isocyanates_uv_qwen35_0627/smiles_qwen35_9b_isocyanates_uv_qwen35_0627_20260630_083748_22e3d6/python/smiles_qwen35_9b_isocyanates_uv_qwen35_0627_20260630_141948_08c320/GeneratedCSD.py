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
        d_1_steps_: int
        d_1_steps_ = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate one valid isocyanate SMILES. Isocyanates contain N=C=O. Examples: CCN=C=O, c1ccccc1N=C=O. Output only the SMILES string.")))
        d_2_maxPreamble_: int
        d_2_maxPreamble_ = 8
        d_3_preambleCount_: int
        d_3_preambleCount_ = 0
        with _dafny.label("0"):
            while (((d_1_steps_) < (maxSteps)) and ((d_3_preambleCount_) < (d_2_maxPreamble_))) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_4_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_4_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    d_3_preambleCount_ = (d_3_preambleCount_) + (1)
                    if (d_4_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                        if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_5_og_: _dafny.Seq
            d_6_oi_: bool
            d_7_oc_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_5_og_ = out1_
            d_6_oi_ = out2_
            d_7_oc_ = out3_
            generated = d_5_og_
            insideConstrainedOut = d_6_oi_
            currentConstrainedOut = d_7_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        d_8_minSpanTokens_: int
        d_8_minSpanTokens_ = 8
        d_9_spanTokens_: int
        d_9_spanTokens_ = 0
        with _dafny.label("1"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    if (d_9_spanTokens_) >= (d_8_minSpanTokens_):
                        d_10_cg_: _dafny.Seq
                        d_11_ci_: bool
                        d_12_cc_: _dafny.Seq
                        d_13_closed_: bool
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out7_: bool
                        out4_, out5_, out6_, out7_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_10_cg_ = out4_
                        d_11_ci_ = out5_
                        d_12_cc_ = out6_
                        d_13_closed_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_13_closed_:
                            generated = d_10_cg_
                            insideConstrainedOut = d_11_ci_
                            currentConstrainedOut = d_12_cc_
                        elif (d_1_steps_) < (maxSteps):
                            d_14_constrainedPrompt_: _dafny.Seq
                            d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_15_next_: _dafny.Seq
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_15_next_ = out8_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_15_next_) == (eosToken):
                                raise _dafny.Break("1")
                            elif True:
                                d_16_isComplete_: bool
                                d_16_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if not(d_16_isComplete_):
                                    d_17_ag_: _dafny.Seq
                                    d_18_ai_: bool
                                    d_19_ac_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                    d_17_ag_ = out9_
                                    d_18_ai_ = out10_
                                    d_19_ac_ = out11_
                                    generated = d_17_ag_
                                    insideConstrainedOut = d_18_ai_
                                    currentConstrainedOut = d_19_ac_
                                    d_9_spanTokens_ = (d_9_spanTokens_) + (1)
                    elif True:
                        d_20_constrainedPrompt_: _dafny.Seq
                        d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_21_next_: _dafny.Seq
                        out12_: _dafny.Seq
                        out12_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_21_next_ = out12_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_21_next_) == (eosToken):
                            raise _dafny.Break("1")
                        elif True:
                            d_22_isComplete_: bool
                            d_22_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if not(d_22_isComplete_):
                                d_23_ag_: _dafny.Seq
                                d_24_ai_: bool
                                d_25_ac_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                d_23_ag_ = out13_
                                d_24_ai_ = out14_
                                d_25_ac_ = out15_
                                generated = d_23_ag_
                                insideConstrainedOut = d_24_ai_
                                currentConstrainedOut = d_25_ac_
                                d_9_spanTokens_ = (d_9_spanTokens_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_26_closeBudget_: int
            d_26_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_27_cg_: _dafny.Seq
            d_28_ci_: bool
            d_29_cc_: _dafny.Seq
            out16_: _dafny.Seq
            out17_: bool
            out18_: _dafny.Seq
            out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_26_closeBudget_)
            d_27_cg_ = out16_
            d_28_ci_ = out17_
            d_29_cc_ = out18_
            generated = d_27_cg_
            insideConstrainedOut = d_28_ci_
            currentConstrainedOut = d_29_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

