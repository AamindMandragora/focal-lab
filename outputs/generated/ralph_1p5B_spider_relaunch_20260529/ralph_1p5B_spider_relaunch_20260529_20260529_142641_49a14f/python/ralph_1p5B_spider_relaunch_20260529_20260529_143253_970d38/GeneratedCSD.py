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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one SQL query. Output format must be: SQL: <<your SQL query here>>. Use only tables and columns from the schema. Write correct SQL with proper JOINs, GROUP BY, ORDER BY as needed. No explanation.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_2_next1_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_2_next1_ = out0_
            d_1_steps_ = (d_1_steps_) + (1)
            if (d_2_next1_) != (eosToken):
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next1_]))
        if ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_3_next2_: _dafny.Seq
            out1_: _dafny.Seq
            out1_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_3_next2_ = out1_
            d_1_steps_ = (d_1_steps_) + (1)
            if (d_3_next2_) != (eosToken):
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next2_]))
        if ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_4_next3_: _dafny.Seq
            out2_: _dafny.Seq
            out2_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_4_next3_ = out2_
            d_1_steps_ = (d_1_steps_) + (1)
            if (d_4_next3_) != (eosToken):
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next3_]))
        if ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_5_openedGenerated_: _dafny.Seq
            d_6_openedInside_: bool
            d_7_openedCurrent_: _dafny.Seq
            out3_: _dafny.Seq
            out4_: bool
            out5_: _dafny.Seq
            out3_, out4_, out5_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_5_openedGenerated_ = out3_
            d_6_openedInside_ = out4_
            d_7_openedCurrent_ = out5_
            generated = d_5_openedGenerated_
            insideConstrainedOut = d_6_openedInside_
            currentConstrainedOut = d_7_openedCurrent_
            d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_8_closedGenerated_: _dafny.Seq
                        d_9_closedInside_: bool
                        d_10_closedCurrent_: _dafny.Seq
                        out6_: _dafny.Seq
                        out7_: bool
                        out8_: _dafny.Seq
                        out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_8_closedGenerated_ = out6_
                        d_9_closedInside_ = out7_
                        d_10_closedCurrent_ = out8_
                        generated = d_8_closedGenerated_
                        insideConstrainedOut = d_9_closedInside_
                        currentConstrainedOut = d_10_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_11_constrainedPrompt_: _dafny.Seq
                        d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_12_queryLen_: int
                        d_12_queryLen_ = len(currentConstrainedOut)
                        d_13_next_: _dafny.Seq
                        d_13_next_ = eosToken
                        if (d_12_queryLen_) < (3):
                            d_14_sqlStartGroups_: _dafny.Seq
                            d_14_sqlStartGroups_ = (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SELECT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "select")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WITH")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "with"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "DISTINCT")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "distinct")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ALL")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "all"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "from"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "WHERE")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "where"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "JOIN")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "join")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ON")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "on"))])])) + (validTokenGroups)
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, d_14_sqlStartGroups_, _dafny.BigRational('8e0'), eosToken)
                            d_13_next_ = out9_
                        elif (d_12_queryLen_) < (100):
                            d_15_nextCG_: _dafny.Seq
                            d_16_wasCG_: bool
                            out10_: _dafny.Seq
                            out11_: bool
                            out10_, out11_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_15_nextCG_ = out10_
                            d_16_wasCG_ = out11_
                            d_13_next_ = d_15_nextCG_
                        elif True:
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_13_next_ = out12_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_13_next_) == (eosToken):
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_17_closedGenerated_: _dafny.Seq
                                d_18_closedInside_: bool
                                d_19_closedCurrent_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_17_closedGenerated_ = out13_
                                d_18_closedInside_ = out14_
                                d_19_closedCurrent_ = out15_
                                generated = d_17_closedGenerated_
                                insideConstrainedOut = d_18_closedInside_
                                currentConstrainedOut = d_19_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_20_appendedGenerated_: _dafny.Seq
                            d_21_appendedInside_: bool
                            d_22_appendedCurrent_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                            d_20_appendedGenerated_ = out16_
                            d_21_appendedInside_ = out17_
                            d_22_appendedCurrent_ = out18_
                            generated = d_20_appendedGenerated_
                            insideConstrainedOut = d_21_appendedInside_
                            currentConstrainedOut = d_22_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

