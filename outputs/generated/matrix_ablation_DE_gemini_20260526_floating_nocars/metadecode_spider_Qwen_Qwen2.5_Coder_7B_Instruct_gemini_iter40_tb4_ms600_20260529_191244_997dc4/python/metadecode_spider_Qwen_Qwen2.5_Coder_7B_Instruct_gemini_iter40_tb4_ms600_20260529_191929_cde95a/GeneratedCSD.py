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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one task-appropriate SQL query. Do not use Markdown. Use schema hints from the context when they are relevant to the query."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_sqlKeywordGroups_: _dafny.Seq
        d_2_sqlKeywordGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DISTINCT"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "AND")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OR"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "GROUP")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "BY")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "HAVING")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER"))])])
        d_3_seenFrom_: bool
        d_3_seenFrom_ = False
        d_4_seenWhere_: bool
        d_4_seenWhere_ = False
        d_5_narrowThreshold_: int
        d_5_narrowThreshold_ = 10
        d_6_steps_: int
        d_6_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_6_steps_) < (maxSteps)):
            d_7_chunkBudget_: int
            if ((maxSteps) - (d_6_steps_)) < (2):
                d_7_chunkBudget_ = (maxSteps) - (d_6_steps_)
            elif True:
                d_7_chunkBudget_ = 2
            d_8_generatedOut_: _dafny.Seq
            d_9_stoppedOnOpenSpan_: bool
            d_10_stoppedOnEos_: bool
            d_11_stepsUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_7_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_8_generatedOut_ = out0_
            d_9_stoppedOnOpenSpan_ = out1_
            d_10_stoppedOnEos_ = out2_
            d_11_stepsUsed_ = out3_
            d_6_steps_ = (d_6_steps_) + (d_11_stepsUsed_)
            generated = d_8_generatedOut_
            if not(d_10_stoppedOnEos_):
                if d_9_stoppedOnOpenSpan_:
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                    generated = out4_
                    insideConstrainedOut = out5_
                    currentConstrainedOut = out6_
                elif True:
                    if (d_6_steps_) < (maxSteps):
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        generated = out7_
                        insideConstrainedOut = out8_
                        currentConstrainedOut = out9_
                        d_6_steps_ = (d_6_steps_) + (1)
        with _dafny.label("0"):
            while ((d_6_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if (parser).IsCompletePrefix(currentConstrainedOut):
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        generated = out10_
                        insideConstrainedOut = out11_
                        currentConstrainedOut = out12_
                        d_6_steps_ = (d_6_steps_) + (1)
                    elif True:
                        d_12_constrainedPrompt_: _dafny.Seq
                        d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_13_validCount_: int
                        out13_: int
                        out13_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_13_validCount_ = out13_
                        d_14_next_: _dafny.Seq
                        d_14_next_ = eosToken
                        if (not(d_3_seenFrom_)) and ((d_13_validCount_) <= (d_5_narrowThreshold_)):
                            d_15_groups_: _dafny.Seq
                            d_15_groups_ = (d_2_sqlKeywordGroups_) + (validTokenGroups)
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, d_15_groups_, _dafny.BigRational('6e0'), eosToken)
                            d_14_next_ = out14_
                        elif (d_3_seenFrom_) and (not(d_4_seenWhere_)):
                            out15_: _dafny.Seq
                            out15_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ORDER")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "LIMIT"))]), _dafny.BigRational('5e0'), 8, eosToken)
                            d_14_next_ = out15_
                        elif True:
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_5_narrowThreshold_, eosToken)
                            d_14_next_ = out16_
                        d_6_steps_ = (d_6_steps_) + (1)
                        if (d_14_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            out17_: _dafny.Seq
                            out18_: bool
                            out19_: _dafny.Seq
                            out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                            generated = out17_
                            insideConstrainedOut = out18_
                            currentConstrainedOut = out19_
                            if (d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))):
                                d_3_seenFrom_ = True
                            elif (d_14_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE"))):
                                d_4_seenWhere_ = True
                    pass
            pass
        cost = d_6_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

