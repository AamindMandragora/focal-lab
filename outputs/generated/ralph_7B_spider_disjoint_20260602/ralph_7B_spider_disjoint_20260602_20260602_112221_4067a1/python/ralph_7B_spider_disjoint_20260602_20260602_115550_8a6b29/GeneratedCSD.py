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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You are a SQL expert. Generate a single SQL SELECT query. Rules: write lowercase keywords (select, from, where, join, on, group by, having, order by, limit). Do NOT use table aliases - always write full table.column notation. Use count(*) for counting rows. Use join...on for joins. No semicolons. No markdown. Format: SQL: <<your query here>>")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_phase1Budget_: int
        d_2_phase1Budget_ = 2
        d_3_phase1Steps_: int
        d_3_phase1Steps_ = 0
        while (((d_3_phase1Steps_) < (d_2_phase1Budget_)) and ((d_1_steps_) < (maxSteps))) and (not(insideConstrainedOut)):
            d_4_next_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_4_next_ = out0_
            d_1_steps_ = (d_1_steps_) + (1)
            d_3_phase1Steps_ = (d_3_phase1Steps_) + (1)
            if (d_4_next_) == (eosToken):
                d_3_phase1Steps_ = d_2_phase1Budget_
            elif True:
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                    d_5_obsGenerated_: _dafny.Seq
                    d_6_obsInside_: bool
                    d_7_obsCurrent_: _dafny.Seq
                    out1_: _dafny.Seq
                    out2_: bool
                    out3_: _dafny.Seq
                    out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                    d_5_obsGenerated_ = out1_
                    d_6_obsInside_ = out2_
                    d_7_obsCurrent_ = out3_
                    generated = d_5_obsGenerated_
                    insideConstrainedOut = d_6_obsInside_
                    currentConstrainedOut = d_7_obsCurrent_
                    d_3_phase1Steps_ = d_2_phase1Budget_
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_8_openGenerated_: _dafny.Seq
            d_9_openInside_: bool
            d_10_openCurrent_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_8_openGenerated_ = out4_
            d_9_openInside_ = out5_
            d_10_openCurrent_ = out6_
            generated = d_8_openGenerated_
            insideConstrainedOut = d_9_openInside_
            currentConstrainedOut = d_10_openCurrent_
            d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_11_next_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_11_next_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_11_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_12_closedGenerated_: _dafny.Seq
                        d_13_closedInside_: bool
                        d_14_closedCurrent_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_12_closedGenerated_ = out8_
                        d_13_closedInside_ = out9_
                        d_14_closedCurrent_ = out10_
                        generated = d_12_closedGenerated_
                        insideConstrainedOut = d_13_closedInside_
                        currentConstrainedOut = d_14_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_15_isDeadEnd_: bool
                        out11_: bool
                        out11_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_15_isDeadEnd_ = out11_
                        if d_15_isDeadEnd_:
                            d_16_rolledGenerated_: _dafny.Seq
                            d_17_rolledCurrent_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: _dafny.Seq
                            out12_, out13_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_16_rolledGenerated_ = out12_
                            d_17_rolledCurrent_ = out13_
                            generated = d_16_rolledGenerated_
                            currentConstrainedOut = d_17_rolledCurrent_
                        elif True:
                            d_18_constrainedPrompt_: _dafny.Seq
                            d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_19_next_: _dafny.Seq
                            d_20_wasConstrained_: bool
                            out14_: _dafny.Seq
                            out15_: bool
                            out14_, out15_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_19_next_ = out14_
                            d_20_wasConstrained_ = out15_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_19_next_) == (eosToken):
                                if (not((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
                                    d_21_next2_: _dafny.Seq
                                    out16_: _dafny.Seq
                                    out16_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                    d_21_next2_ = out16_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                    if (d_21_next2_) == (eosToken):
                                        raise _dafny.Break("0")
                                    elif True:
                                        d_22_appendedGenerated_: _dafny.Seq
                                        d_23_appendedInside_: bool
                                        d_24_appendedCurrent_: _dafny.Seq
                                        out17_: _dafny.Seq
                                        out18_: bool
                                        out19_: _dafny.Seq
                                        out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next2_)
                                        d_22_appendedGenerated_ = out17_
                                        d_23_appendedInside_ = out18_
                                        d_24_appendedCurrent_ = out19_
                                        generated = d_22_appendedGenerated_
                                        insideConstrainedOut = d_23_appendedInside_
                                        currentConstrainedOut = d_24_appendedCurrent_
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                d_25_appendedGenerated_: _dafny.Seq
                                d_26_appendedInside_: bool
                                d_27_appendedCurrent_: _dafny.Seq
                                out20_: _dafny.Seq
                                out21_: bool
                                out22_: _dafny.Seq
                                out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                d_25_appendedGenerated_ = out20_
                                d_26_appendedInside_ = out21_
                                d_27_appendedCurrent_ = out22_
                                generated = d_25_appendedGenerated_
                                insideConstrainedOut = d_26_appendedInside_
                                currentConstrainedOut = d_27_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

