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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the word problem with concise reasoning. Do not write << or >> in prose. End with one final symbolic arithmetic expression or number; the decoder will wrap that final expression in visible delimiters. Inside delimiters use only math tokens, variables, numbers, parentheses, and operators."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_mathGroups_: _dafny.Seq
        d_2_mathGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "//")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "."))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "int")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "ceil")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "floor"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "x")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "y")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "z")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "k")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "m")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "d")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w3"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "target")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "sides")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n3"))])])
        d_3_penaltyTokens_: _dafny.Seq
        d_3_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "\n\n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " ")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "  ")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ",")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "?")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "!")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "because")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "therefore")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "is")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "the")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "of")), eosToken])
        d_4_steps_: int
        d_4_steps_ = 0
        d_5_forceAfter_: int
        d_5_forceAfter_ = 60
        d_6_hardStop_: int
        d_6_hardStop_ = 180
        d_7_forcedSpan_: bool
        d_7_forcedSpan_ = insideConstrainedOut
        with _dafny.label("0"):
            while (d_4_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if (d_4_steps_) >= (d_6_hardStop_):
                        if insideConstrainedOut:
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                d_8_stopClosedGenerated_: _dafny.Seq
                                d_9_stopClosedInside_: bool
                                d_10_stopClosedCurrent_: _dafny.Seq
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: _dafny.Seq
                                out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_8_stopClosedGenerated_ = out0_
                                d_9_stopClosedInside_ = out1_
                                d_10_stopClosedCurrent_ = out2_
                                generated = d_8_stopClosedGenerated_
                                insideConstrainedOut = d_9_stopClosedInside_
                                currentConstrainedOut = d_10_stopClosedCurrent_
                                d_4_steps_ = (d_4_steps_) + (1)
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_4_steps_ = (d_4_steps_) + (1)
                        raise _dafny.Break("0")
                    elif not(insideConstrainedOut):
                        if (not(d_7_forcedSpan_)) and ((d_4_steps_) >= (d_5_forceAfter_)):
                            d_11_openedGenerated_: _dafny.Seq
                            d_12_openedInside_: bool
                            d_13_openedCurrent_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_11_openedGenerated_ = out3_
                            d_12_openedInside_ = out4_
                            d_13_openedCurrent_ = out5_
                            generated = d_11_openedGenerated_
                            insideConstrainedOut = d_12_openedInside_
                            currentConstrainedOut = d_13_openedCurrent_
                            d_7_forcedSpan_ = True
                            d_4_steps_ = (d_4_steps_) + (1)
                        elif True:
                            d_14_nextFree_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_14_nextFree_ = out6_
                            d_4_steps_ = (d_4_steps_) + (1)
                            if (d_14_nextFree_) == (eosToken):
                                if (not(d_7_forcedSpan_)) and ((d_4_steps_) < (maxSteps)):
                                    d_15_eosOpenedGenerated_: _dafny.Seq
                                    d_16_eosOpenedInside_: bool
                                    d_17_eosOpenedCurrent_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_15_eosOpenedGenerated_ = out7_
                                    d_16_eosOpenedInside_ = out8_
                                    d_17_eosOpenedCurrent_ = out9_
                                    generated = d_15_eosOpenedGenerated_
                                    insideConstrainedOut = d_16_eosOpenedInside_
                                    currentConstrainedOut = d_17_eosOpenedCurrent_
                                    d_7_forcedSpan_ = True
                                    d_4_steps_ = (d_4_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif (d_14_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                pass
                            elif (d_14_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                pass
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_14_nextFree_]))
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_18_closedGenerated_: _dafny.Seq
                        d_19_closedInside_: bool
                        d_20_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_18_closedGenerated_ = out10_
                        d_19_closedInside_ = out11_
                        d_20_closedCurrent_ = out12_
                        generated = d_18_closedGenerated_
                        insideConstrainedOut = d_19_closedInside_
                        currentConstrainedOut = d_20_closedCurrent_
                        d_4_steps_ = (d_4_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_21_constrainedPrompt_: _dafny.Seq
                        d_21_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_22_combinedGroups_: _dafny.Seq
                        d_22_combinedGroups_ = (d_2_mathGroups_) + (validTokenGroups)
                        d_23_nextConstrained_: _dafny.Seq
                        out13_: _dafny.Seq
                        out13_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, d_22_combinedGroups_, _dafny.BigRational('4e0'), d_3_penaltyTokens_, _dafny.BigRational('1e1'), 12, eosToken)
                        d_23_nextConstrained_ = out13_
                        d_4_steps_ = (d_4_steps_) + (1)
                        if (d_23_nextConstrained_) == (eosToken):
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_4_steps_) < (maxSteps)):
                                d_24_eosClosedGenerated_: _dafny.Seq
                                d_25_eosClosedInside_: bool
                                d_26_eosClosedCurrent_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_24_eosClosedGenerated_ = out14_
                                d_25_eosClosedInside_ = out15_
                                d_26_eosClosedCurrent_ = out16_
                                generated = d_24_eosClosedGenerated_
                                insideConstrainedOut = d_25_eosClosedInside_
                                currentConstrainedOut = d_26_eosClosedCurrent_
                                d_4_steps_ = (d_4_steps_) + (1)
                            elif (d_4_steps_) < (maxSteps):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_4_steps_ = (d_4_steps_) + (1)
                            raise _dafny.Break("0")
                        elif (d_23_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_23_nextConstrained_]))
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            raise _dafny.Break("0")
                        elif True:
                            d_27_validNext_: bool
                            out17_: bool
                            out17_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_23_nextConstrained_)
                            d_27_validNext_ = out17_
                            if d_27_validNext_:
                                d_28_appendedGenerated_: _dafny.Seq
                                d_29_appendedInside_: bool
                                d_30_appendedCurrent_: _dafny.Seq
                                out18_: _dafny.Seq
                                out19_: bool
                                out20_: _dafny.Seq
                                out18_, out19_, out20_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_nextConstrained_)
                                d_28_appendedGenerated_ = out18_
                                d_29_appendedInside_ = out19_
                                d_30_appendedCurrent_ = out20_
                                generated = d_28_appendedGenerated_
                                insideConstrainedOut = d_29_appendedInside_
                                currentConstrainedOut = d_30_appendedCurrent_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_4_steps_) < (maxSteps)):
                                    d_31_nowClosedGenerated_: _dafny.Seq
                                    d_32_nowClosedInside_: bool
                                    d_33_nowClosedCurrent_: _dafny.Seq
                                    out21_: _dafny.Seq
                                    out22_: bool
                                    out23_: _dafny.Seq
                                    out21_, out22_, out23_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_31_nowClosedGenerated_ = out21_
                                    d_32_nowClosedInside_ = out22_
                                    d_33_nowClosedCurrent_ = out23_
                                    generated = d_31_nowClosedGenerated_
                                    insideConstrainedOut = d_32_nowClosedInside_
                                    currentConstrainedOut = d_33_nowClosedCurrent_
                                    d_4_steps_ = (d_4_steps_) + (1)
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_4_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

