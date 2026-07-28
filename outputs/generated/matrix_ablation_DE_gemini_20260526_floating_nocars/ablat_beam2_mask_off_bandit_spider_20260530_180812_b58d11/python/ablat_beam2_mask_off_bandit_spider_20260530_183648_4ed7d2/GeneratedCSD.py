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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one SQLite query using only the provided schema. Output exactly SQL: <<YOUR QUERY>> with no explanation, Markdown, or extra text. Prefer the model's best semantic SQL, avoid unnecessary joins, avoid inventing columns, and use table/column names exactly as in the schema."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Treat contextual token groups as schema hints, but use them only when they are relevant to the question.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_headerStage_: int
        d_2_headerStage_ = 0
        d_3_eosPenaltyTokens_: _dafny.Seq
        d_3_eosPenaltyTokens_ = _dafny.SeqWithoutIsStrInference([eosToken, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))])
        d_4_steps_: int
        d_4_steps_ = 0
        with _dafny.label("0"):
            while (d_4_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_2_headerStage_) == (0):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:"))]))
                            d_2_headerStage_ = 1
                            d_4_steps_ = (d_4_steps_) + (1)
                        elif (d_2_headerStage_) == (1):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " "))]))
                            d_2_headerStage_ = 2
                            d_4_steps_ = (d_4_steps_) + (1)
                        elif True:
                            d_5_openedGenerated_: _dafny.Seq
                            d_6_openedInside_: bool
                            d_7_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_openedGenerated_ = out0_
                            d_6_openedInside_ = out1_
                            d_7_openedCurrent_ = out2_
                            generated = d_5_openedGenerated_
                            insideConstrainedOut = d_6_openedInside_
                            currentConstrainedOut = d_7_openedCurrent_
                            d_4_steps_ = (d_4_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_8_closedGenerated_: _dafny.Seq
                        d_9_closedInside_: bool
                        d_10_closedCurrent_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_8_closedGenerated_ = out3_
                        d_9_closedInside_ = out4_
                        d_10_closedCurrent_ = out5_
                        generated = d_8_closedGenerated_
                        insideConstrainedOut = d_9_closedInside_
                        currentConstrainedOut = d_10_closedCurrent_
                        d_4_steps_ = (d_4_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_11_constrainedPrompt_: _dafny.Seq
                        d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_12_next_: _dafny.Seq
                        d_13_wasConstrained_: bool
                        out6_: _dafny.Seq
                        out7_: bool
                        out6_, out7_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_12_next_ = out6_
                        d_13_wasConstrained_ = out7_
                        d_4_steps_ = (d_4_steps_) + (1)
                        if (d_12_next_) == (eosToken):
                            if (d_4_steps_) < (maxSteps):
                                d_14_next2_: _dafny.Seq
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, d_3_eosPenaltyTokens_, _dafny.BigRational('8e0'), eosToken)
                                d_14_next2_ = out8_
                                d_4_steps_ = (d_4_steps_) + (1)
                                if (d_14_next2_) != (eosToken):
                                    d_15_valid2_: bool
                                    out9_: bool
                                    out9_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_14_next2_)
                                    d_15_valid2_ = out9_
                                    if d_15_valid2_:
                                        d_16_appendedGenerated2_: _dafny.Seq
                                        d_17_appendedInside2_: bool
                                        d_18_appendedCurrent2_: _dafny.Seq
                                        out10_: _dafny.Seq
                                        out11_: bool
                                        out12_: _dafny.Seq
                                        out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next2_)
                                        d_16_appendedGenerated2_ = out10_
                                        d_17_appendedInside2_ = out11_
                                        d_18_appendedCurrent2_ = out12_
                                        generated = d_16_appendedGenerated2_
                                        insideConstrainedOut = d_17_appendedInside2_
                                        currentConstrainedOut = d_18_appendedCurrent2_
                        elif True:
                            d_19_valid_: bool
                            out13_: bool
                            out13_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_12_next_)
                            d_19_valid_ = out13_
                            if d_19_valid_:
                                d_20_appendedGenerated_: _dafny.Seq
                                d_21_appendedInside_: bool
                                d_22_appendedCurrent_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                                d_20_appendedGenerated_ = out14_
                                d_21_appendedInside_ = out15_
                                d_22_appendedCurrent_ = out16_
                                generated = d_20_appendedGenerated_
                                insideConstrainedOut = d_21_appendedInside_
                                currentConstrainedOut = d_22_appendedCurrent_
                    pass
            pass
        cost = d_4_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

