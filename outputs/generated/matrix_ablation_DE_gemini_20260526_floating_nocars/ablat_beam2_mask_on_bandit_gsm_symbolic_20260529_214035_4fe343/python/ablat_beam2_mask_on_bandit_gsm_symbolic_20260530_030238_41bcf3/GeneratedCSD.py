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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Wrap each important arithmetic expression and the final answer inside visible << >> delimiters. Inside delimiters write only a compact symbolic arithmetic expression or number, not prose.")))
        if (maxSteps) == (0):
            cost = 0
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        d_1_operatorGroups_: _dafny.Seq
        d_1_operatorGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "//")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "."))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "int")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ceil")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "floor")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "%"))])])
        d_2_penaltyTokens_: _dafny.Seq
        d_2_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "{")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "}")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "?")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "if")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "else")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "then")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "for")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "in")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "and")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "or")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "because")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore"))])
        d_3_steps_: int
        d_3_steps_ = 0
        d_4_forceAfter_: int
        d_4_forceAfter_ = 120
        d_5_forcedSpan_: bool
        d_5_forcedSpan_ = insideConstrainedOut
        d_6_done_: bool
        d_6_done_ = False
        while ((d_3_steps_) < (maxSteps)) and (not(d_6_done_)):
            if not(insideConstrainedOut):
                if (not(d_5_forcedSpan_)) and ((d_3_steps_) >= (d_4_forceAfter_)):
                    d_7_openedGenerated_: _dafny.Seq
                    d_8_openedInside_: bool
                    d_9_openedCurrent_: _dafny.Seq
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: _dafny.Seq
                    out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_7_openedGenerated_ = out0_
                    d_8_openedInside_ = out1_
                    d_9_openedCurrent_ = out2_
                    generated = d_7_openedGenerated_
                    insideConstrainedOut = d_8_openedInside_
                    currentConstrainedOut = d_9_openedCurrent_
                    d_5_forcedSpan_ = True
                    d_3_steps_ = (d_3_steps_) + (1)
                elif True:
                    d_10_nextFree_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_10_nextFree_ = out3_
                    d_3_steps_ = (d_3_steps_) + (1)
                    if (d_10_nextFree_) == (eosToken):
                        if (not(d_5_forcedSpan_)) and ((d_3_steps_) < (maxSteps)):
                            d_11_eosOpenedGenerated_: _dafny.Seq
                            d_12_eosOpenedInside_: bool
                            d_13_eosOpenedCurrent_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_11_eosOpenedGenerated_ = out4_
                            d_12_eosOpenedInside_ = out5_
                            d_13_eosOpenedCurrent_ = out6_
                            generated = d_11_eosOpenedGenerated_
                            insideConstrainedOut = d_12_eosOpenedInside_
                            currentConstrainedOut = d_13_eosOpenedCurrent_
                            d_5_forcedSpan_ = True
                            d_3_steps_ = (d_3_steps_) + (1)
                        elif True:
                            d_6_done_ = True
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_10_nextFree_]))
                        if (d_10_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_5_forcedSpan_ = True
            elif (parser).IsCompletePrefix(currentConstrainedOut):
                d_14_closedGenerated_: _dafny.Seq
                d_15_closedInside_: bool
                d_16_closedCurrent_: _dafny.Seq
                out7_: _dafny.Seq
                out8_: bool
                out9_: _dafny.Seq
                out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_14_closedGenerated_ = out7_
                d_15_closedInside_ = out8_
                d_16_closedCurrent_ = out9_
                generated = d_14_closedGenerated_
                insideConstrainedOut = d_15_closedInside_
                currentConstrainedOut = d_16_closedCurrent_
                d_3_steps_ = (d_3_steps_) + (1)
                d_6_done_ = True
            elif True:
                d_17_constrainedPrompt_: _dafny.Seq
                d_17_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_18_combinedGroups_: _dafny.Seq
                d_18_combinedGroups_ = (validTokenGroups) + (d_1_operatorGroups_)
                d_19_nextConstrained_: _dafny.Seq
                out10_: _dafny.Seq
                out10_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_17_constrainedPrompt_, currentConstrainedOut, d_18_combinedGroups_, _dafny.BigRational('6e0'), d_2_penaltyTokens_, _dafny.BigRational('6e0'), 16, eosToken)
                d_19_nextConstrained_ = out10_
                d_3_steps_ = (d_3_steps_) + (1)
                if (d_19_nextConstrained_) == (eosToken):
                    d_6_done_ = True
                elif True:
                    d_20_validNext_: bool
                    out11_: bool
                    out11_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_19_nextConstrained_)
                    d_20_validNext_ = out11_
                    if d_20_validNext_:
                        d_21_appendedGenerated_: _dafny.Seq
                        d_22_appendedInside_: bool
                        d_23_appendedCurrent_: _dafny.Seq
                        out12_: _dafny.Seq
                        out13_: bool
                        out14_: _dafny.Seq
                        out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_nextConstrained_)
                        d_21_appendedGenerated_ = out12_
                        d_22_appendedInside_ = out13_
                        d_23_appendedCurrent_ = out14_
                        generated = d_21_appendedGenerated_
                        insideConstrainedOut = d_22_appendedInside_
                        currentConstrainedOut = d_23_appendedCurrent_
        cost = d_3_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

