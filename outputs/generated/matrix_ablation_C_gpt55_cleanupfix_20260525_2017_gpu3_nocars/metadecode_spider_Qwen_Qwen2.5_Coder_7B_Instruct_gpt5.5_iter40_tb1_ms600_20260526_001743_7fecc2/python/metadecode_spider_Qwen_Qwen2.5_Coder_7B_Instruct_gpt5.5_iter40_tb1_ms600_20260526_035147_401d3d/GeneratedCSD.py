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
        if (maxSteps) == (0):
            cost = 0
        elif (maxSteps) == (1):
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:"))]))
            cost = 1
        elif (maxSteps) == (2):
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " "))]))
            cost = 2
        elif True:
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " "))]))
            d_1_openedGenerated_: _dafny.Seq
            d_2_openedInside_: bool
            d_3_openedCurrent_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_1_openedGenerated_ = out0_
            d_2_openedInside_ = out1_
            d_3_openedCurrent_ = out2_
            generated = d_1_openedGenerated_
            insideConstrainedOut = d_2_openedInside_
            currentConstrainedOut = d_3_openedCurrent_
            d_4_steps_: int
            d_4_steps_ = 3
            d_5_hitEos_: bool
            d_5_hitEos_ = False
            d_6_allowDistinct_: bool
            d_6_allowDistinct_ = ((((((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "distinct"))) in (prompt)) or ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Distinct"))) in (prompt))) or ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "different"))) in (prompt))) or ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Different"))) in (prompt))) or ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "unique"))) in (prompt))) or ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Unique"))) in (prompt))) or ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "duplicates"))) in (prompt))) or ((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "duplicate"))) in (prompt))
            d_7_penaltyTokens_: _dafny.Seq
            d_7_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "T5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AS")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "as"))])
            if not(d_6_allowDistinct_):
                d_7_penaltyTokens_ = (d_7_penaltyTokens_) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DISTINCT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "distinct"))]))
            d_8_guidedTokenLimit_: int
            d_8_guidedTokenLimit_ = 90
            while ((((((d_4_steps_) + (1)) < (maxSteps)) and (insideConstrainedOut)) and ((len(currentConstrainedOut)) < (d_8_guidedTokenLimit_))) and (not(d_5_hitEos_))) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                d_9_stablePrefix_: _dafny.Seq
                d_9_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_10_constrainedPrompt_: _dafny.Seq
                d_10_constrainedPrompt_ = (prompt) + (d_9_stablePrefix_)
                d_11_next_: _dafny.Seq
                out3_: _dafny.Seq
                out3_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), d_7_penaltyTokens_, _dafny.BigRational('18e0'), 24, eosToken)
                d_11_next_ = out3_
                d_4_steps_ = (d_4_steps_) + (1)
                if (d_11_next_) == (eosToken):
                    d_5_hitEos_ = True
                elif True:
                    d_12_appendedGenerated_: _dafny.Seq
                    d_13_appendedInside_: bool
                    d_14_appendedCurrent_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                    d_12_appendedGenerated_ = out4_
                    d_13_appendedInside_ = out5_
                    d_14_appendedCurrent_ = out6_
                    generated = d_12_appendedGenerated_
                    insideConstrainedOut = d_13_appendedInside_
                    currentConstrainedOut = d_14_appendedCurrent_
            if (((((d_4_steps_) + (1)) < (maxSteps)) and (insideConstrainedOut)) and (not(d_5_hitEos_))) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                d_15_firstCap_: int
                d_15_firstCap_ = ((maxSteps) - (d_4_steps_)) - (1)
                if (d_15_firstCap_) > (150):
                    d_15_firstCap_ = 150
                d_16_stablePrefix2_: _dafny.Seq
                d_16_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_17_constrainedPrompt2_: _dafny.Seq
                d_17_constrainedPrompt2_ = (prompt) + (d_16_stablePrefix2_)
                d_18_symbolGenerated_: _dafny.Seq
                d_19_symbolOut_: _dafny.Seq
                d_20_symbolHitEos_: bool
                d_21_used_: int
                out7_: _dafny.Seq
                out8_: _dafny.Seq
                out9_: bool
                out10_: int
                out7_, out8_, out9_, out10_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_17_constrainedPrompt2_, generated, currentConstrainedOut, d_15_firstCap_, eosToken)
                d_18_symbolGenerated_ = out7_
                d_19_symbolOut_ = out8_
                d_20_symbolHitEos_ = out9_
                d_21_used_ = out10_
                generated = d_18_symbolGenerated_
                currentConstrainedOut = d_19_symbolOut_
                insideConstrainedOut = True
                d_5_hitEos_ = d_20_symbolHitEos_
                d_4_steps_ = (d_4_steps_) + (d_21_used_)
            while ((((((d_4_steps_) + (1)) < (maxSteps)) and (insideConstrainedOut)) and ((len(currentConstrainedOut)) < (260))) and (not(d_5_hitEos_))) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                d_22_stablePrefix3_: _dafny.Seq
                d_22_stablePrefix3_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                d_23_constrainedPrompt3_: _dafny.Seq
                d_23_constrainedPrompt3_ = (prompt) + (d_22_stablePrefix3_)
                d_24_next2_: _dafny.Seq
                out11_: _dafny.Seq
                out11_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_23_constrainedPrompt3_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('5e0'), d_7_penaltyTokens_, _dafny.BigRational('14e0'), 18, eosToken)
                d_24_next2_ = out11_
                d_4_steps_ = (d_4_steps_) + (1)
                if (d_24_next2_) == (eosToken):
                    d_5_hitEos_ = True
                elif True:
                    d_25_appendedGenerated2_: _dafny.Seq
                    d_26_appendedInside2_: bool
                    d_27_appendedCurrent2_: _dafny.Seq
                    out12_: _dafny.Seq
                    out13_: bool
                    out14_: _dafny.Seq
                    out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_next2_)
                    d_25_appendedGenerated2_ = out12_
                    d_26_appendedInside2_ = out13_
                    d_27_appendedCurrent2_ = out14_
                    generated = d_25_appendedGenerated2_
                    insideConstrainedOut = d_26_appendedInside2_
                    currentConstrainedOut = d_27_appendedCurrent2_
            if ((insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_4_steps_) < (maxSteps)):
                d_28_closedGenerated_: _dafny.Seq
                d_29_closedInside_: bool
                d_30_closedCurrent_: _dafny.Seq
                out15_: _dafny.Seq
                out16_: bool
                out17_: _dafny.Seq
                out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_28_closedGenerated_ = out15_
                d_29_closedInside_ = out16_
                d_30_closedCurrent_ = out17_
                generated = d_28_closedGenerated_
                insideConstrainedOut = d_29_closedInside_
                currentConstrainedOut = d_30_closedCurrent_
                d_4_steps_ = (d_4_steps_) + (1)
            cost = d_4_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

