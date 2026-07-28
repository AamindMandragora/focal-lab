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
        d_2_spanLen_: int
        d_2_spanLen_ = 0
        d_3_minSpanLen_: int
        d_3_minSpanLen_ = 5
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a SMILES string for an isocyanate molecule containing N=C=O. The SMILES must include the isocyanate group: N=C=O. Valid examples: CCN=C=O, CCCN=C=O, CCCCN=C=O, c1ccccc1N=C=O, CC(C)N=C=O, C1CCCCC1N=C=O, c1ccccc1CN=C=O, CC(C)(C)N=C=O. The N=C=O group is required. Output only the SMILES string with no other text.")))
        d_4_nGroup_: _dafny.Seq
        d_4_nGroup_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N"))])
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
                                d_2_spanLen_ = 0
                    elif True:
                        d_6_nCount_: int
                        out1_: int
                        out1_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N")))
                        d_6_nCount_ = out1_
                        d_7_hasN_: bool
                        d_7_hasN_ = (d_6_nCount_) > (0)
                        if ((d_7_hasN_) and ((d_2_spanLen_) >= (d_3_minSpanLen_))) or ((d_2_spanLen_) >= (20)):
                            d_8_closeBudget_: int
                            d_8_closeBudget_ = (maxSteps) - (d_1_steps_)
                            d_9_cg_: _dafny.Seq
                            d_10_ci_: bool
                            d_11_cc_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_8_closeBudget_)
                            d_9_cg_ = out2_
                            d_10_ci_ = out3_
                            d_11_cc_ = out4_
                            generated = d_9_cg_
                            insideConstrainedOut = d_10_ci_
                            currentConstrainedOut = d_11_cc_
                            d_1_steps_ = maxSteps
                        elif True:
                            d_12_constrainedPrompt_: _dafny.Seq
                            d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_13_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            if (not(d_7_hasN_)) and ((d_2_spanLen_) >= (1)):
                                out5_: _dafny.Seq
                                out5_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, d_4_nGroup_, _dafny.BigRational('8e0'), eosToken)
                                d_13_next_ = out5_
                            elif (not(d_7_hasN_)) and ((d_2_spanLen_) == (0)):
                                out6_: _dafny.Seq
                                out6_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('11e-1'), eosToken)
                                d_13_next_ = out6_
                            elif True:
                                out7_: _dafny.Seq
                                out7_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                                d_13_next_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_13_next_) == (eosToken):
                                if (parser).IsCompletePrefix(currentConstrainedOut):
                                    d_14_closedGenerated_: _dafny.Seq
                                    d_15_closedInside_: bool
                                    d_16_closedCurrent_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out9_: bool
                                    out10_: _dafny.Seq
                                    out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_14_closedGenerated_ = out8_
                                    d_15_closedInside_ = out9_
                                    d_16_closedCurrent_ = out10_
                                    generated = d_14_closedGenerated_
                                    insideConstrainedOut = d_15_closedInside_
                                    currentConstrainedOut = d_16_closedCurrent_
                                    d_2_spanLen_ = 0
                                raise _dafny.Break("0")
                            elif True:
                                d_17_isComplete_: bool
                                d_17_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if d_17_isComplete_:
                                    d_18_closedGenerated_: _dafny.Seq
                                    d_19_closedInside_: bool
                                    d_20_closedCurrent_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_18_closedGenerated_ = out11_
                                    d_19_closedInside_ = out12_
                                    d_20_closedCurrent_ = out13_
                                    generated = d_18_closedGenerated_
                                    insideConstrainedOut = d_19_closedInside_
                                    currentConstrainedOut = d_20_closedCurrent_
                                    d_2_spanLen_ = 0
                                elif True:
                                    d_21_appendedGenerated_: _dafny.Seq
                                    d_22_appendedInside_: bool
                                    d_23_appendedCurrent_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                                    d_21_appendedGenerated_ = out14_
                                    d_22_appendedInside_ = out15_
                                    d_23_appendedCurrent_ = out16_
                                    generated = d_21_appendedGenerated_
                                    insideConstrainedOut = d_22_appendedInside_
                                    currentConstrainedOut = d_23_appendedCurrent_
                                    d_2_spanLen_ = (d_2_spanLen_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

