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
        insideConstrainedOut = False
        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = 0
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step, but be concise. Use visible delimiters exactly like <<expression>> for every intermediate symbolic arithmetic expression and for the final answer. Inside << >> put only a short algebraic expression or number: no words, no units, no LaTeX, no nested <<, and close the span immediately. End with a line like Final answer: <<final_expression>>."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Prefer the variables and arithmetic operator tokens from the problem when writing expressions.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_localMax_: int
        d_2_localMax_ = maxSteps
        if (d_2_localMax_) > (220):
            d_2_localMax_ = 220
        d_3_rawInside_: bool
        d_3_rawInside_ = insideConstrained
        d_4_spanLen_: int
        d_4_spanLen_ = 0
        d_5_spanTokenLimit_: int
        d_5_spanTokenLimit_ = 15
        d_6_proseSinceSpan_: int
        d_6_proseSinceSpan_ = 0
        d_7_firstOpenLimit_: int
        d_7_firstOpenLimit_ = 48
        d_8_completeSpans_: int
        d_8_completeSpans_ = 0
        d_9_sawFinal_: bool
        d_9_sawFinal_ = False
        d_10_steps_: int
        d_10_steps_ = 0
        with _dafny.label("0"):
            while ((d_10_steps_) < (maxSteps)) and ((d_10_steps_) < (d_2_localMax_)):
                with _dafny.c_label("0"):
                    if (d_3_rawInside_) and ((((d_4_spanLen_) >= (d_5_spanTokenLimit_)) or (((d_10_steps_) + (1)) == (maxSteps))) or (((d_10_steps_) + (1)) == (d_2_localMax_))):
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                        d_3_rawInside_ = False
                        d_4_spanLen_ = 0
                        d_8_completeSpans_ = (d_8_completeSpans_) + (1)
                        d_10_steps_ = (d_10_steps_) + (1)
                        if d_9_sawFinal_:
                            raise _dafny.Break("0")
                    elif ((((not(d_3_rawInside_)) and ((d_8_completeSpans_) == (0))) and ((d_6_proseSinceSpan_) >= (d_7_firstOpenLimit_))) and (((d_10_steps_) + (1)) < (maxSteps))) and (((d_10_steps_) + (1)) < (d_2_localMax_)):
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
                        d_3_rawInside_ = True
                        d_4_spanLen_ = 0
                        d_6_proseSinceSpan_ = 0
                        d_10_steps_ = (d_10_steps_) + (1)
                    elif True:
                        d_11_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_11_next_ = out0_
                        d_10_steps_ = (d_10_steps_) + (1)
                        if (d_11_next_) == (eosToken):
                            if ((d_3_rawInside_) and ((d_10_steps_) < (maxSteps))) and ((d_10_steps_) < (d_2_localMax_)):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                                d_3_rawInside_ = False
                                d_4_spanLen_ = 0
                                d_8_completeSpans_ = (d_8_completeSpans_) + (1)
                                d_10_steps_ = (d_10_steps_) + (1)
                            raise _dafny.Break("0")
                        elif d_3_rawInside_:
                            if (d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_4_spanLen_ = d_4_spanLen_
                            elif (d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                d_3_rawInside_ = False
                                d_4_spanLen_ = 0
                                d_8_completeSpans_ = (d_8_completeSpans_) + (1)
                                if d_9_sawFinal_:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                d_4_spanLen_ = (d_4_spanLen_) + (1)
                        elif True:
                            if (d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                d_6_proseSinceSpan_ = (d_6_proseSinceSpan_) + (1)
                            elif (d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                if ((d_10_steps_) < (maxSteps)) and ((d_10_steps_) < (d_2_localMax_)):
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                    d_3_rawInside_ = True
                                    d_4_spanLen_ = 0
                                    d_6_proseSinceSpan_ = 0
                                elif True:
                                    d_6_proseSinceSpan_ = (d_6_proseSinceSpan_) + (1)
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                d_6_proseSinceSpan_ = (d_6_proseSinceSpan_) + (1)
                                if ((((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Final")))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "final"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Answer"))))) or ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))):
                                    d_9_sawFinal_ = True
                                if ((d_8_completeSpans_) >= (3)) and ((d_6_proseSinceSpan_) >= (24)):
                                    raise _dafny.Break("0")
                    pass
            pass
        cost = d_10_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

