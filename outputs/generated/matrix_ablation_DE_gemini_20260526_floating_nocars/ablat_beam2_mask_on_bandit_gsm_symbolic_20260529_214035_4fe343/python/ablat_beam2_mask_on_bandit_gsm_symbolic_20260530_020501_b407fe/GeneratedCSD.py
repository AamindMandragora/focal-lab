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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the word problem step by step in concise prose. Put calculations and the final answer in visible << >> spans. Inside a span write only a compact arithmetic expression or number: no words, no if-clauses, no extra delimiters."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_mathGroups_: _dafny.Seq
        d_2_mathGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "//")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "."))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "int")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ceil")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "floor"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "x")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "y")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "d")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "r")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "k")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "m")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "target")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "frac_1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "frac_2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sides")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "name")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "obj"))])])
        d_3_combinedGroups_: _dafny.Seq
        d_3_combinedGroups_ = (d_2_mathGroups_) + (validTokenGroups)
        d_4_penaltyTokens_: _dafny.Seq
        d_4_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>>>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\t")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "?")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "if")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "else")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "then")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "and")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "or")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "because")), eosToken])
        d_5_steps_: int
        d_5_steps_ = 0
        d_6_forceAfter_: int
        d_6_forceAfter_ = 80
        d_7_forcedSpan_: bool
        d_7_forcedSpan_ = insideConstrainedOut
        d_8_localLimit_: int
        d_8_localLimit_ = 180
        if (maxSteps) < (d_8_localLimit_):
            d_8_localLimit_ = maxSteps
        with _dafny.label("0"):
            while ((d_5_steps_) < (maxSteps)) and ((d_5_steps_) < (d_8_localLimit_)):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_7_forcedSpan_)) and ((d_5_steps_) >= (d_6_forceAfter_)):
                            d_9_openedGenerated_: _dafny.Seq
                            d_10_openedInside_: bool
                            d_11_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_9_openedGenerated_ = out0_
                            d_10_openedInside_ = out1_
                            d_11_openedCurrent_ = out2_
                            generated = d_9_openedGenerated_
                            insideConstrainedOut = d_10_openedInside_
                            currentConstrainedOut = d_11_openedCurrent_
                            d_7_forcedSpan_ = True
                            d_5_steps_ = (d_5_steps_) + (1)
                        elif True:
                            d_12_nextFree_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_12_nextFree_ = out3_
                            d_5_steps_ = (d_5_steps_) + (1)
                            if (d_12_nextFree_) == (eosToken):
                                if d_7_forcedSpan_:
                                    raise _dafny.Break("0")
                            elif (((d_12_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))) or ((d_12_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">"))))) or ((d_12_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>>>")))):
                                pass
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_nextFree_]))
                                if (d_12_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_13_enteredGenerated_: _dafny.Seq
                                    d_14_enteredInside_: bool
                                    d_15_enteredCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_13_enteredGenerated_ = out4_
                                    d_14_enteredInside_ = out5_
                                    d_15_enteredCurrent_ = out6_
                                    generated = d_13_enteredGenerated_
                                    insideConstrainedOut = d_14_enteredInside_
                                    currentConstrainedOut = d_15_enteredCurrent_
                                    d_7_forcedSpan_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_16_closedGenerated_: _dafny.Seq
                        d_17_closedInside_: bool
                        d_18_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_16_closedGenerated_ = out7_
                        d_17_closedInside_ = out8_
                        d_18_closedCurrent_ = out9_
                        generated = d_16_closedGenerated_
                        insideConstrainedOut = d_17_closedInside_
                        currentConstrainedOut = d_18_closedCurrent_
                        d_5_steps_ = (d_5_steps_) + (1)
                    elif True:
                        d_19_constrainedPrompt_: _dafny.Seq
                        d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_20_nextConstrained_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, d_3_combinedGroups_, _dafny.BigRational('5e0'), d_4_penaltyTokens_, _dafny.BigRational('7e0'), 12, eosToken)
                        d_20_nextConstrained_ = out10_
                        d_5_steps_ = (d_5_steps_) + (1)
                        if (d_20_nextConstrained_) == (eosToken):
                            raise _dafny.Break("0")
                        elif (((((((((((((d_20_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or ((d_20_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))))) or ((d_20_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<"))))) or ((d_20_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">"))))) or ((d_20_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>>>"))))) or ((d_20_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " "))))) or ((d_20_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\t"))))) or ((d_20_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n"))))) or ((d_20_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n\n"))))) or ((d_20_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "if"))))) or ((d_20_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "else"))))) or ((d_20_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "then"))))) or ((d_20_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "because")))):
                            pass
                        elif True:
                            d_21_validNext_: bool
                            out11_: bool
                            out11_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_20_nextConstrained_)
                            d_21_validNext_ = out11_
                            if d_21_validNext_:
                                d_22_appendedGenerated_: _dafny.Seq
                                d_23_appendedInside_: bool
                                d_24_appendedCurrent_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_nextConstrained_)
                                d_22_appendedGenerated_ = out12_
                                d_23_appendedInside_ = out13_
                                d_24_appendedCurrent_ = out14_
                                generated = d_22_appendedGenerated_
                                insideConstrainedOut = d_23_appendedInside_
                                currentConstrainedOut = d_24_appendedCurrent_
                    pass
            pass
        cost = d_5_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

