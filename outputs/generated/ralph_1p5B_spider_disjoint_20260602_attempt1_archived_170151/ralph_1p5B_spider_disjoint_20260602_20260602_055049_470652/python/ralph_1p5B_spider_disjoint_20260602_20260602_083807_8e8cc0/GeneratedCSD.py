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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one SQL query in the format: SQL: <<SELECT ...>>. Use only tables and columns from the provided schema. Do not add explanation or markdown. Write minimal correct SQL. Do not list all columns in GROUP BY - only group by the columns that appear in SELECT. Use simple JOINs only when necessary."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_3_chunkBudget_: int
            d_3_chunkBudget_ = 6
            if (d_3_chunkBudget_) > ((maxSteps) - (d_2_steps_)):
                d_3_chunkBudget_ = (maxSteps) - (d_2_steps_)
            if (d_3_chunkBudget_) > (0):
                d_4_chunkGenerated_: _dafny.Seq
                d_5_stoppedOnOpen_: bool
                d_6_stoppedOnEos_: bool
                d_7_chunkSteps_: int
                out0_: _dafny.Seq
                out1_: bool
                out2_: bool
                out3_: int
                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                d_4_chunkGenerated_ = out0_
                d_5_stoppedOnOpen_ = out1_
                d_6_stoppedOnEos_ = out2_
                d_7_chunkSteps_ = out3_
                generated = d_4_chunkGenerated_
                d_2_steps_ = (d_2_steps_) + (d_7_chunkSteps_)
                if d_6_stoppedOnEos_:
                    cost = d_2_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                if d_5_stoppedOnOpen_:
                    d_8_eg_: _dafny.Seq
                    d_9_ei_: bool
                    d_10_ec_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                    d_8_eg_ = out4_
                    d_9_ei_ = out5_
                    d_10_ec_ = out6_
                    generated = d_8_eg_
                    insideConstrainedOut = d_9_ei_
                    currentConstrainedOut = d_10_ec_
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_11_eg_: _dafny.Seq
            d_12_ei_: bool
            d_13_ec_: _dafny.Seq
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_11_eg_ = out7_
            d_12_ei_ = out8_
            d_13_ec_ = out9_
            generated = d_11_eg_
            insideConstrainedOut = d_12_ei_
            currentConstrainedOut = d_13_ec_
            d_2_steps_ = (d_2_steps_) + (1)
        d_14_spanTokensUsed_: int
        d_14_spanTokensUsed_ = 0
        d_15_spanTokenBudget_: int
        d_15_spanTokenBudget_ = 120
        d_16_useAdaptive_: bool
        d_16_useAdaptive_ = True
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_17_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_17_next_ = out10_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_17_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_17_next_]))
                            if (d_17_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_18_eg_: _dafny.Seq
                                d_19_ei_: bool
                                d_20_ec_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_18_eg_ = out11_
                                d_19_ei_ = out12_
                                d_20_ec_ = out13_
                                generated = d_18_eg_
                                insideConstrainedOut = d_19_ei_
                                currentConstrainedOut = d_20_ec_
                                d_14_spanTokensUsed_ = 0
                                d_16_useAdaptive_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_21_closedGenerated_: _dafny.Seq
                        d_22_closedInside_: bool
                        d_23_closedCurrent_: _dafny.Seq
                        out14_: _dafny.Seq
                        out15_: bool
                        out16_: _dafny.Seq
                        out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_21_closedGenerated_ = out14_
                        d_22_closedInside_ = out15_
                        d_23_closedCurrent_ = out16_
                        generated = d_21_closedGenerated_
                        insideConstrainedOut = d_22_closedInside_
                        currentConstrainedOut = d_23_closedCurrent_
                        d_2_steps_ = (d_2_steps_) + (1)
                    elif True:
                        d_24_constrainedPrompt_: _dafny.Seq
                        d_24_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_25_isDeadEnd_: bool
                        out17_: bool
                        out17_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_25_isDeadEnd_ = out17_
                        if d_25_isDeadEnd_:
                            raise _dafny.Break("0")
                        d_26_next_: _dafny.Seq
                        d_26_next_ = eosToken
                        if (d_16_useAdaptive_) and ((d_14_spanTokensUsed_) < (20)):
                            out18_: _dafny.Seq
                            out18_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 16, eosToken)
                            d_26_next_ = out18_
                        elif True:
                            d_27_nextCG_: _dafny.Seq
                            d_28_wasConstrained_: bool
                            out19_: _dafny.Seq
                            out20_: bool
                            out19_, out20_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_24_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_27_nextCG_ = out19_
                            d_28_wasConstrained_ = out20_
                            d_26_next_ = d_27_nextCG_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_26_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_29_appendedGenerated_: _dafny.Seq
                            d_30_appendedInside_: bool
                            d_31_appendedCurrent_: _dafny.Seq
                            out21_: _dafny.Seq
                            out22_: bool
                            out23_: _dafny.Seq
                            out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_next_)
                            d_29_appendedGenerated_ = out21_
                            d_30_appendedInside_ = out22_
                            d_31_appendedCurrent_ = out23_
                            generated = d_29_appendedGenerated_
                            insideConstrainedOut = d_30_appendedInside_
                            currentConstrainedOut = d_31_appendedCurrent_
                            d_14_spanTokensUsed_ = (d_14_spanTokensUsed_) + (1)
                            if (d_14_spanTokensUsed_) >= (20):
                                d_16_useAdaptive_ = False
                            if (d_14_spanTokensUsed_) >= (d_15_spanTokenBudget_):
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

