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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Return exactly one visible constrained span containing a valid SMILES string for the requested molecular class. Begin the span promptly, keep every constrained prefix parser-valid, avoid empty spans, and close the span as soon as the SMILES is complete.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_openedGenerated_: _dafny.Seq
                        d_3_openedInside_: bool
                        d_4_openedCurrent_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_2_openedGenerated_ = out0_
                        d_3_openedInside_ = out1_
                        d_4_openedCurrent_ = out2_
                        generated = d_2_openedGenerated_
                        insideConstrainedOut = d_3_openedInside_
                        currentConstrainedOut = d_4_openedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_5_closedGenerated_: _dafny.Seq
                        d_6_closedInside_: bool
                        d_7_closedCurrent_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_5_closedGenerated_ = out3_
                        d_6_closedInside_ = out4_
                        d_7_closedCurrent_ = out5_
                        generated = d_5_closedGenerated_
                        insideConstrainedOut = d_6_closedInside_
                        currentConstrainedOut = d_7_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_8_stablePrefix_: _dafny.Seq
                        d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_9_constrainedPrompt_: _dafny.Seq
                        d_9_constrainedPrompt_ = (prompt) + (d_8_stablePrefix_)
                        d_10_validCount_: int
                        out6_: int
                        out6_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_10_validCount_ = out6_
                        if (len(currentConstrainedOut)) < (2):
                            d_11_candidates_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, 4, eosToken)
                            d_11_candidates_ = out7_
                            d_12_next_: _dafny.Seq
                            d_12_next_ = (d_11_candidates_)[0]
                            if ((d_12_next_) == (eosToken)) and ((len(d_11_candidates_)) > (1)):
                                d_12_next_ = (d_11_candidates_)[1]
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_12_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_13_appendedGenerated0_: _dafny.Seq
                                d_14_appendedInside0_: bool
                                d_15_appendedCurrent0_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                                d_13_appendedGenerated0_ = out8_
                                d_14_appendedInside0_ = out9_
                                d_15_appendedCurrent0_ = out10_
                                generated = d_13_appendedGenerated0_
                                insideConstrainedOut = d_14_appendedInside0_
                                currentConstrainedOut = d_15_appendedCurrent0_
                        elif (d_10_validCount_) <= (6):
                            d_16_nextTight_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_16_nextTight_ = out11_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_16_nextTight_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_17_appendedGenerated1_: _dafny.Seq
                                d_18_appendedInside1_: bool
                                d_19_appendedCurrent1_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_nextTight_)
                                d_17_appendedGenerated1_ = out12_
                                d_18_appendedInside1_ = out13_
                                d_19_appendedCurrent1_ = out14_
                                generated = d_17_appendedGenerated1_
                                insideConstrainedOut = d_18_appendedInside1_
                                currentConstrainedOut = d_19_appendedCurrent1_
                        elif (len(validTokenGroups)) > (0):
                            d_20_nextGroup_: _dafny.Seq
                            out15_: _dafny.Seq
                            out15_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_20_nextGroup_ = out15_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_20_nextGroup_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_21_appendedGenerated2_: _dafny.Seq
                                d_22_appendedInside2_: bool
                                d_23_appendedCurrent2_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_nextGroup_)
                                d_21_appendedGenerated2_ = out16_
                                d_22_appendedInside2_ = out17_
                                d_23_appendedCurrent2_ = out18_
                                generated = d_21_appendedGenerated2_
                                insideConstrainedOut = d_22_appendedInside2_
                                currentConstrainedOut = d_23_appendedCurrent2_
                        elif True:
                            d_24_nextSoft_: _dafny.Seq
                            out19_: _dafny.Seq
                            out19_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('8e-1'), eosToken)
                            d_24_nextSoft_ = out19_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_24_nextSoft_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_25_appendedGenerated3_: _dafny.Seq
                                d_26_appendedInside3_: bool
                                d_27_appendedCurrent3_: _dafny.Seq
                                out20_: _dafny.Seq
                                out21_: bool
                                out22_: _dafny.Seq
                                out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_24_nextSoft_)
                                d_25_appendedGenerated3_ = out20_
                                d_26_appendedInside3_ = out21_
                                d_27_appendedCurrent3_ = out22_
                                generated = d_25_appendedGenerated3_
                                insideConstrainedOut = d_26_appendedInside3_
                                currentConstrainedOut = d_27_appendedCurrent3_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

