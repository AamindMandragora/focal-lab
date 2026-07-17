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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step in concise prose. Wrap useful intermediate symbolic expressions and the final answer in visible << >> delimiters. Inside each delimiter, write only a compact arithmetic expression or number, with no words. Always close a delimiter immediately after the expression; never write another << inside an open span."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_mathGroups_: _dafny.Seq
        d_2_mathGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "//")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "."))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "int")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ceil")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "floor"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "x")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "y")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "d")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "r")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "k")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "m")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "target")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n_1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n_2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n_3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n_4"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "frac_1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "frac_2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "k_2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "k_3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sides"))])])
        d_3_combinedGroups_: _dafny.Seq
        d_3_combinedGroups_ = (d_2_mathGroups_) + (validTokenGroups)
        d_4_penaltyTokens_: _dafny.Seq
        d_4_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "?")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "if")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "else")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "for")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "while")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "return")), eosToken])
        d_5_steps_: int
        d_5_steps_ = 0
        d_6_forceAfter_: int
        d_6_forceAfter_ = 80
        d_7_forcedSpan_: bool
        d_7_forcedSpan_ = insideConstrainedOut
        with _dafny.label("0"):
            while (d_5_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_7_forcedSpan_)) and ((d_5_steps_) >= (d_6_forceAfter_)):
                            d_8_openedGenerated_: _dafny.Seq
                            d_9_openedInside_: bool
                            d_10_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_8_openedGenerated_ = out0_
                            d_9_openedInside_ = out1_
                            d_10_openedCurrent_ = out2_
                            generated = d_8_openedGenerated_
                            insideConstrainedOut = d_9_openedInside_
                            currentConstrainedOut = d_10_openedCurrent_
                            d_7_forcedSpan_ = True
                            d_5_steps_ = (d_5_steps_) + (1)
                        elif True:
                            d_11_nextFree_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_11_nextFree_ = out3_
                            d_5_steps_ = (d_5_steps_) + (1)
                            if (d_11_nextFree_) == (eosToken):
                                raise _dafny.Break("0")
                            elif (d_11_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_nextFree_]))
                                d_12_enteredGenerated_: _dafny.Seq
                                d_13_enteredInside_: bool
                                d_14_enteredCurrent_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_12_enteredGenerated_ = out4_
                                d_13_enteredInside_ = out5_
                                d_14_enteredCurrent_ = out6_
                                generated = d_12_enteredGenerated_
                                insideConstrainedOut = d_13_enteredInside_
                                currentConstrainedOut = d_14_enteredCurrent_
                                d_7_forcedSpan_ = True
                            elif (d_11_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                pass
                            elif (d_11_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<"))):
                                pass
                            elif (d_11_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">"))):
                                pass
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_nextFree_]))
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_15_closedGenerated_: _dafny.Seq
                        d_16_closedInside_: bool
                        d_17_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_15_closedGenerated_ = out7_
                        d_16_closedInside_ = out8_
                        d_17_closedCurrent_ = out9_
                        generated = d_15_closedGenerated_
                        insideConstrainedOut = d_16_closedInside_
                        currentConstrainedOut = d_17_closedCurrent_
                        d_5_steps_ = (d_5_steps_) + (1)
                    elif True:
                        d_18_closeByLength_: bool
                        d_18_closeByLength_ = False
                        if (len(currentConstrainedOut)) >= (32):
                            d_19_lastForLength_: _dafny.Seq
                            d_19_lastForLength_ = (currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]
                            d_18_closeByLength_ = (((((((((((d_19_lastForLength_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")))) and ((d_19_lastForLength_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-"))))) and ((d_19_lastForLength_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*"))))) and ((d_19_lastForLength_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/"))))) and ((d_19_lastForLength_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "//"))))) and ((d_19_lastForLength_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "("))))) and ((d_19_lastForLength_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ","))))) and ((d_19_lastForLength_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))))) and ((d_19_lastForLength_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))))) and ((d_19_lastForLength_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<"))))) and ((d_19_lastForLength_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">"))))
                        if d_18_closeByLength_:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_5_steps_ = (d_5_steps_) + (1)
                        elif True:
                            d_20_constrainedPrompt_: _dafny.Seq
                            d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_21_nextConstrained_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, d_3_combinedGroups_, _dafny.BigRational('5e0'), d_4_penaltyTokens_, _dafny.BigRational('2e1'), 12, eosToken)
                            d_21_nextConstrained_ = out10_
                            d_5_steps_ = (d_5_steps_) + (1)
                            if (d_21_nextConstrained_) == (eosToken):
                                d_22_canCloseOnEos_: bool
                                d_22_canCloseOnEos_ = False
                                if (len(currentConstrainedOut)) > (0):
                                    d_23_lastOnEos_: _dafny.Seq
                                    d_23_lastOnEos_ = (currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]
                                    d_22_canCloseOnEos_ = (((((((((((d_23_lastOnEos_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")))) and ((d_23_lastOnEos_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-"))))) and ((d_23_lastOnEos_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*"))))) and ((d_23_lastOnEos_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/"))))) and ((d_23_lastOnEos_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "//"))))) and ((d_23_lastOnEos_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "("))))) and ((d_23_lastOnEos_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ","))))) and ((d_23_lastOnEos_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))))) and ((d_23_lastOnEos_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))))) and ((d_23_lastOnEos_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<"))))) and ((d_23_lastOnEos_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">"))))
                                if d_22_canCloseOnEos_:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                elif True:
                                    raise _dafny.Break("0")
                            elif ((((d_21_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or ((d_21_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<"))))) or ((d_21_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">"))))) or ((d_21_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))):
                                d_24_canCloseOnDelimiter_: bool
                                d_24_canCloseOnDelimiter_ = False
                                if (len(currentConstrainedOut)) > (0):
                                    d_25_lastOnDelimiter_: _dafny.Seq
                                    d_25_lastOnDelimiter_ = (currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]
                                    d_24_canCloseOnDelimiter_ = (((((((((((d_25_lastOnDelimiter_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")))) and ((d_25_lastOnDelimiter_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-"))))) and ((d_25_lastOnDelimiter_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*"))))) and ((d_25_lastOnDelimiter_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/"))))) and ((d_25_lastOnDelimiter_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "//"))))) and ((d_25_lastOnDelimiter_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "("))))) and ((d_25_lastOnDelimiter_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ","))))) and ((d_25_lastOnDelimiter_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))))) and ((d_25_lastOnDelimiter_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))))) and ((d_25_lastOnDelimiter_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<"))))) and ((d_25_lastOnDelimiter_) != (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">"))))
                                if d_24_canCloseOnDelimiter_:
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            elif True:
                                d_26_validNext_: bool
                                out11_: bool
                                out11_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_21_nextConstrained_)
                                d_26_validNext_ = out11_
                                if d_26_validNext_:
                                    d_27_appendedGenerated_: _dafny.Seq
                                    d_28_appendedInside_: bool
                                    d_29_appendedCurrent_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out13_: bool
                                    out14_: _dafny.Seq
                                    out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_nextConstrained_)
                                    d_27_appendedGenerated_ = out12_
                                    d_28_appendedInside_ = out13_
                                    d_29_appendedCurrent_ = out14_
                                    generated = d_27_appendedGenerated_
                                    insideConstrainedOut = d_28_appendedInside_
                                    currentConstrainedOut = d_29_appendedCurrent_
                    pass
            pass
        cost = d_5_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

