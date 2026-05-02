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
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokens, eosToken):
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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_fromKeyword_: _dafny.Seq
        d_1_fromKeyword_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        d_3_semanticContext_: _dafny.Seq
        d_3_semanticContext_ = _dafny.SeqWithoutIsStrInference([])
        d_4_steps_: int
        d_4_steps_ = 0
        with _dafny.label("0"):
            while (d_4_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_5_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_5_next_ = out0_
                        d_4_steps_ = (d_4_steps_) + (1)
                        if (d_5_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                            if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        out1_: _dafny.Seq
                        out1_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(currentConstrainedOut, d_1_fromKeyword_)
                        d_3_semanticContext_ = out1_
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_6_closedGenerated_: _dafny.Seq
                            d_7_closedInside_: bool
                            d_8_closedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_6_closedGenerated_ = out2_
                            d_7_closedInside_ = out3_
                            d_8_closedCurrent_ = out4_
                            generated = d_6_closedGenerated_
                            insideConstrainedOut = d_7_closedInside_
                            currentConstrainedOut = d_8_closedCurrent_
                            d_4_steps_ = (d_4_steps_) + (1)
                        elif True:
                            d_9_constrainedPrompt_: _dafny.Seq
                            d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_10_narrow_: bool
                            out5_: bool
                            out5_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, d_2_narrowThreshold_)
                            d_10_narrow_ = out5_
                            if d_10_narrow_:
                                d_11_next_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_11_next_ = out6_
                                d_4_steps_ = (d_4_steps_) + (1)
                                if (d_11_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_12_appendedGenerated_: _dafny.Seq
                                    d_13_appendedInside_: bool
                                    d_14_appendedCurrent_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                                    d_12_appendedGenerated_ = out7_
                                    d_13_appendedInside_ = out8_
                                    d_14_appendedCurrent_ = out9_
                                    generated = d_12_appendedGenerated_
                                    insideConstrainedOut = d_13_appendedInside_
                                    currentConstrainedOut = d_14_appendedCurrent_
                            elif True:
                                (lm).GenerateLogits((d_9_constrainedPrompt_) + (currentConstrainedOut))
                                d_15_candidates_: _dafny.Seq
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, 24, eosToken)
                                d_15_candidates_ = out10_
                                if (len(validTokens)) > (0):
                                    d_16_preferred_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out11_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_15_candidates_, validTokens)
                                    d_16_preferred_ = out11_
                                    if (len(d_16_preferred_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_16_preferred_, _dafny.BigRational('5e0'))
                                if (len(d_3_semanticContext_)) > (0):
                                    d_17_scoped_: _dafny.Seq
                                    out12_: _dafny.Seq
                                    out12_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_15_candidates_, d_3_semanticContext_)
                                    d_17_scoped_ = out12_
                                    if (len(d_17_scoped_)) > (0):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_17_scoped_, _dafny.BigRational('6e0'))
                                (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                                d_18_next_: _dafny.Seq
                                out13_: _dafny.Seq
                                out13_ = (lm).ChooseNextToken()
                                d_18_next_ = out13_
                                (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                                d_4_steps_ = (d_4_steps_) + (1)
                                if (d_18_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_19_appendedGenerated2_: _dafny.Seq
                                    d_20_appendedInside2_: bool
                                    d_21_appendedCurrent2_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_18_next_)
                                    d_19_appendedGenerated2_ = out14_
                                    d_20_appendedInside2_ = out15_
                                    d_21_appendedCurrent2_ = out16_
                                    generated = d_19_appendedGenerated2_
                                    insideConstrainedOut = d_20_appendedInside2_
                                    currentConstrainedOut = d_21_appendedCurrent2_
                    pass
            pass
        cost = d_4_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

