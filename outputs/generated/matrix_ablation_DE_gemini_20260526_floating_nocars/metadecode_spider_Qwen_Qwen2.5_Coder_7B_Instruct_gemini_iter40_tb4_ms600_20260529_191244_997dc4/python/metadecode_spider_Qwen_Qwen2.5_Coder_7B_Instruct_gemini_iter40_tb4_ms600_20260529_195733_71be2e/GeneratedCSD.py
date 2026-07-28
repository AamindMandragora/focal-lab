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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one task-appropriate SQL query in the format `SQL: <<query>>`. Do not use Markdown. Use schema hints from the context when they are relevant to the query."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_3_chunkBudget_: int
            if ((maxSteps) - (d_2_steps_)) < (8):
                d_3_chunkBudget_ = (maxSteps) - (d_2_steps_)
            elif True:
                d_3_chunkBudget_ = 8
            d_4_generatedOut_: _dafny.Seq
            d_5_stoppedOnOpenSpan_: bool
            d_6_stoppedOnEos_: bool
            d_7_stepsUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_4_generatedOut_ = out0_
            d_5_stoppedOnOpenSpan_ = out1_
            d_6_stoppedOnEos_ = out2_
            d_7_stepsUsed_ = out3_
            d_2_steps_ = (d_2_steps_) + (d_7_stepsUsed_)
            generated = d_4_generatedOut_
            if not(d_6_stoppedOnEos_):
                if d_5_stoppedOnOpenSpan_:
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                    generated = out4_
                    insideConstrainedOut = out5_
                    currentConstrainedOut = out6_
                elif True:
                    if (d_2_steps_) < (maxSteps):
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        generated = out7_
                        insideConstrainedOut = out8_
                        currentConstrainedOut = out9_
                        d_2_steps_ = (d_2_steps_) + (1)
        with _dafny.label("0"):
            while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if (parser).IsCompletePrefix(currentConstrainedOut):
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        generated = out10_
                        insideConstrainedOut = out11_
                        currentConstrainedOut = out12_
                        d_2_steps_ = (d_2_steps_) + (1)
                    elif True:
                        (lm).GenerateLogits((prompt) + (generated))
                        d_8_topToken_: _dafny.Seq
                        out13_: _dafny.Seq
                        out13_ = (lm).ChooseNextTokenUnconstrained()
                        d_8_topToken_ = out13_
                        d_9_isTopTokenValid_: bool = False
                        if (d_8_topToken_) == (eosToken):
                            d_9_isTopTokenValid_ = True
                        elif True:
                            out14_: bool
                            out14_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_8_topToken_)
                            d_9_isTopTokenValid_ = out14_
                        d_10_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if d_9_isTopTokenValid_:
                            d_10_next_ = d_8_topToken_
                        elif True:
                            (d_0_helpers_).BoostValidGroups(lm, parser, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'))
                            (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                            out15_: _dafny.Seq
                            out15_ = (lm).ChooseNextToken()
                            d_10_next_ = out15_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_10_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                            generated = out16_
                            insideConstrainedOut = out17_
                            currentConstrainedOut = out18_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

