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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one novel acrylate SMILES string. An acrylate must contain the vinyl ester core C=CC(=O)O. Valid examples: C=CC(=O)OCCC, C=CC(=O)OCCCC, C=CC(=O)OCC(C)C, C=C(C)C(=O)OCCC, C=CC(=O)OCCCCC, C=CC(=O)OCC(CC)CC, C=CC(=O)OCCCCCC, C=CC(=O)OCCOC. Output only the SMILES string.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_minConstrainedTokens_: int
        d_2_minConstrainedTokens_ = 12
        d_3_maxPreamble_: int
        d_3_maxPreamble_ = 40
        d_4_preambleSteps_: int
        d_4_preambleSteps_ = 0
        with _dafny.label("0"):
            while (((d_1_steps_) < (maxSteps)) and ((d_4_preambleSteps_) < (d_3_maxPreamble_))) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_5_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_5_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    d_4_preambleSteps_ = (d_4_preambleSteps_) + (1)
                    if (d_5_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                        if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_6_og_: _dafny.Seq
            d_7_oi_: bool
            d_8_oc_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_6_og_ = out1_
            d_7_oi_ = out2_
            d_8_oc_ = out3_
            generated = d_6_og_
            insideConstrainedOut = d_7_oi_
            currentConstrainedOut = d_8_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("1"):
            while (((d_1_steps_) < (maxSteps)) and (insideConstrainedOut)) and ((len(currentConstrainedOut)) < (d_2_minConstrainedTokens_)):
                with _dafny.c_label("1"):
                    d_9_constrainedPrompt_: _dafny.Seq
                    d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_10_next_: _dafny.Seq
                    out4_: _dafny.Seq
                    out4_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('2e0'), _dafny.SeqWithoutIsStrInference([]), _dafny.BigRational('1e0'), 3, eosToken)
                    d_10_next_ = out4_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_10_next_) == (eosToken):
                        raise _dafny.Break("1")
                    elif True:
                        d_11_isComplete_: bool
                        d_11_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if not(d_11_isComplete_):
                            d_12_ag_: _dafny.Seq
                            d_13_ai_: bool
                            d_14_ac_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                            d_12_ag_ = out5_
                            d_13_ai_ = out6_
                            d_14_ac_ = out7_
                            generated = d_12_ag_
                            insideConstrainedOut = d_13_ai_
                            currentConstrainedOut = d_14_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_15_closeBudget_: int
            d_15_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_16_cg_: _dafny.Seq
            d_17_ci_: bool
            d_18_cc_: _dafny.Seq
            out8_: _dafny.Seq
            out9_: bool
            out10_: _dafny.Seq
            out8_, out9_, out10_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_15_closeBudget_)
            d_16_cg_ = out8_
            d_17_ci_ = out9_
            d_18_cc_ = out10_
            generated = d_16_cg_
            insideConstrainedOut = d_17_ci_
            currentConstrainedOut = d_18_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

