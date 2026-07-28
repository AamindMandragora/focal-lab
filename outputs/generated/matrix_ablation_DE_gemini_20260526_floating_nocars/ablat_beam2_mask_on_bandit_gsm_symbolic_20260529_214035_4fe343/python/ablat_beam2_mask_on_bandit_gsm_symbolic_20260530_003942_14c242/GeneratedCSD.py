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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step in concise prose. Wrap useful intermediate symbolic expressions and the final answer in visible << >> delimiters. Inside each delimiter, write only a compact arithmetic expression or number, with no words."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_mathGroups_: _dafny.Seq
        d_2_mathGroups_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "+")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "-")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "*")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "//")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "/")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "."))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "int"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "x")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "y")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "t")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "d")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "r")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "k")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "m")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "total")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "w4"))]), _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n_1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n_2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n_3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n_4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "k_2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "k_3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "p1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "p2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "p3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "p4"))])])
        d_3_combinedGroups_: _dafny.Seq
        d_3_combinedGroups_ = (d_2_mathGroups_) + (validTokenGroups)
        if (maxSteps) == (0):
            cost = 0
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        d_4_steps_: int
        d_4_steps_ = 0
        d_5_done_: bool
        d_5_done_ = False
        if not(insideConstrainedOut):
            d_6_openedGenerated_: _dafny.Seq
            d_7_openedInside_: bool
            d_8_openedCurrent_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_6_openedGenerated_ = out0_
            d_7_openedInside_ = out1_
            d_8_openedCurrent_ = out2_
            generated = d_6_openedGenerated_
            insideConstrainedOut = d_7_openedInside_
            currentConstrainedOut = d_8_openedCurrent_
            d_4_steps_ = (d_4_steps_) + (1)
        elif (parser).IsCompletePrefix(currentConstrainedOut):
            d_9_closedGenerated_: _dafny.Seq
            d_10_closedInside_: bool
            d_11_closedCurrent_: _dafny.Seq
            out3_: _dafny.Seq
            out4_: bool
            out5_: _dafny.Seq
            out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
            d_9_closedGenerated_ = out3_
            d_10_closedInside_ = out4_
            d_11_closedCurrent_ = out5_
            generated = d_9_closedGenerated_
            insideConstrainedOut = d_10_closedInside_
            currentConstrainedOut = d_11_closedCurrent_
            d_4_steps_ = (d_4_steps_) + (1)
            d_5_done_ = True
        elif True:
            d_12_constrainedPrompt0_: _dafny.Seq
            d_12_constrainedPrompt0_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
            d_13_nextConstrained0_: _dafny.Seq
            out6_: _dafny.Seq
            out6_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_12_constrainedPrompt0_, currentConstrainedOut, d_3_combinedGroups_, _dafny.BigRational('4e0'), 12, eosToken)
            d_13_nextConstrained0_ = out6_
            d_4_steps_ = (d_4_steps_) + (1)
            if (d_13_nextConstrained0_) == (eosToken):
                d_5_done_ = True
            elif (d_13_nextConstrained0_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                pass
            elif (d_13_nextConstrained0_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                pass
            elif True:
                d_14_validNext0_: bool
                out7_: bool
                out7_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_13_nextConstrained0_)
                d_14_validNext0_ = out7_
                if d_14_validNext0_:
                    d_15_appendedGenerated0_: _dafny.Seq
                    d_16_appendedInside0_: bool
                    d_17_appendedCurrent0_: _dafny.Seq
                    out8_: _dafny.Seq
                    out9_: bool
                    out10_: _dafny.Seq
                    out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_nextConstrained0_)
                    d_15_appendedGenerated0_ = out8_
                    d_16_appendedInside0_ = out9_
                    d_17_appendedCurrent0_ = out10_
                    generated = d_15_appendedGenerated0_
                    insideConstrainedOut = d_16_appendedInside0_
                    currentConstrainedOut = d_17_appendedCurrent0_
        while ((d_4_steps_) < (maxSteps)) and (not(d_5_done_)):
            if not(insideConstrainedOut):
                d_18_nextFree_: _dafny.Seq
                out11_: _dafny.Seq
                out11_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                d_18_nextFree_ = out11_
                d_4_steps_ = (d_4_steps_) + (1)
                if (d_18_nextFree_) == (eosToken):
                    d_5_done_ = True
                elif (d_18_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_18_nextFree_]))
                    insideConstrainedOut = True
                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                elif (d_18_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                    pass
                elif True:
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_18_nextFree_]))
            elif (parser).IsCompletePrefix(currentConstrainedOut):
                d_19_closedGeneratedLoop_: _dafny.Seq
                d_20_closedInsideLoop_: bool
                d_21_closedCurrentLoop_: _dafny.Seq
                out12_: _dafny.Seq
                out13_: bool
                out14_: _dafny.Seq
                out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_19_closedGeneratedLoop_ = out12_
                d_20_closedInsideLoop_ = out13_
                d_21_closedCurrentLoop_ = out14_
                generated = d_19_closedGeneratedLoop_
                insideConstrainedOut = d_20_closedInsideLoop_
                currentConstrainedOut = d_21_closedCurrentLoop_
                d_4_steps_ = (d_4_steps_) + (1)
                d_5_done_ = True
            elif True:
                d_22_constrainedPrompt_: _dafny.Seq
                d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                d_23_nextConstrained_: _dafny.Seq
                out15_: _dafny.Seq
                out15_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, d_3_combinedGroups_, _dafny.BigRational('4e0'), 12, eosToken)
                d_23_nextConstrained_ = out15_
                d_4_steps_ = (d_4_steps_) + (1)
                if (d_23_nextConstrained_) == (eosToken):
                    d_5_done_ = True
                elif (d_23_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                    pass
                elif (d_23_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                    pass
                elif True:
                    d_24_validNext_: bool
                    out16_: bool
                    out16_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_23_nextConstrained_)
                    d_24_validNext_ = out16_
                    if d_24_validNext_:
                        d_25_appendedGenerated_: _dafny.Seq
                        d_26_appendedInside_: bool
                        d_27_appendedCurrent_: _dafny.Seq
                        out17_: _dafny.Seq
                        out18_: bool
                        out19_: _dafny.Seq
                        out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_nextConstrained_)
                        d_25_appendedGenerated_ = out17_
                        d_26_appendedInside_ = out18_
                        d_27_appendedCurrent_ = out19_
                        generated = d_25_appendedGenerated_
                        insideConstrainedOut = d_26_appendedInside_
                        currentConstrainedOut = d_27_appendedCurrent_
        cost = d_4_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

