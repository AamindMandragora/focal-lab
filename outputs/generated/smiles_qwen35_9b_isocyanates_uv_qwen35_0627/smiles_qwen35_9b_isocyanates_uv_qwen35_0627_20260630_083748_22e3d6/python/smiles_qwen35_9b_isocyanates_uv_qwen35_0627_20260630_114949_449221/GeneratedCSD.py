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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate one novel isocyanate SMILES. Isocyanates have N=C=O functional group. Diverse valid examples: CCN=C=O, c1ccccc1N=C=O, CCCN=C=O, CC(C)N=C=O, ClCCN=C=O, O=C=NCCN=C=O, CC(=O)N=C=O, c1ccc(cc1)CN=C=O, CCCCN=C=O, FC(F)(F)CN=C=O. Output only a single SMILES string with N=C=O substructure.")))
        d_2_maxPreamble_: int
        d_2_maxPreamble_ = 5
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
        d_8_buildUpTarget_: int
        d_8_buildUpTarget_ = 40
        d_9_buildCount_: int
        d_9_buildCount_ = 0
        with _dafny.label("1"):
            while (((d_1_steps_) < (maxSteps)) and ((d_9_buildCount_) < (d_8_buildUpTarget_))) and (insideConstrainedOut):
                with _dafny.c_label("1"):
                    d_10_isComplete_: bool
                    d_10_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                    if d_10_isComplete_:
                        raise _dafny.Break("1")
                    d_11_constrainedPrompt_: _dafny.Seq
                    d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_12_penaltyTokens_: _dafny.Seq
                    d_12_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([])
                    d_13_next_: _dafny.Seq
                    out4_: _dafny.Seq
                    out4_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_12_penaltyTokens_, _dafny.BigRational('0e0'), 12, eosToken)
                    d_13_next_ = out4_
                    d_1_steps_ = (d_1_steps_) + (1)
                    d_9_buildCount_ = (d_9_buildCount_) + (1)
                    if (d_13_next_) == (eosToken):
                        raise _dafny.Break("1")
                    elif True:
                        d_14_isCompleteNow_: bool
                        d_14_isCompleteNow_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if not(d_14_isCompleteNow_):
                            d_15_ag_: _dafny.Seq
                            d_16_ai_: bool
                            d_17_ac_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                            d_15_ag_ = out5_
                            d_16_ai_ = out6_
                            d_17_ac_ = out7_
                            generated = d_15_ag_
                            insideConstrainedOut = d_16_ai_
                            currentConstrainedOut = d_17_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_18_closeBudget_: int
            d_18_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_19_cg_: _dafny.Seq
            d_20_ci_: bool
            d_21_cc_: _dafny.Seq
            out8_: _dafny.Seq
            out9_: bool
            out10_: _dafny.Seq
            out8_, out9_, out10_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_18_closeBudget_)
            d_19_cg_ = out8_
            d_20_ci_ = out9_
            d_21_cc_ = out10_
            generated = d_19_cg_
            insideConstrainedOut = d_20_ci_
            currentConstrainedOut = d_21_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

