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
        d_3_seenFrom_: bool
        d_3_seenFrom_ = False
        d_4_i__init_: int
        d_4_i__init_ = 0
        with _dafny.label("0"):
            while (d_4_i__init_) < (len(currentConstrained)):
                with _dafny.c_label("0"):
                    if ((currentConstrained)[d_4_i__init_]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM"))):
                        d_3_seenFrom_ = True
                        raise _dafny.Break("0")
                    d_4_i__init_ = (d_4_i__init_) + (1)
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
            d_5_chunkBudget_: int
            if ((maxSteps) - (d_2_steps_)) < (8):
                d_5_chunkBudget_ = (maxSteps) - (d_2_steps_)
            elif True:
                d_5_chunkBudget_ = 8
            d_6_generatedOut_: _dafny.Seq
            d_7_stoppedOnOpenSpan_: bool
            d_8_stoppedOnEos_: bool
            d_9_stepsUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_6_generatedOut_ = out0_
            d_7_stoppedOnOpenSpan_ = out1_
            d_8_stoppedOnEos_ = out2_
            d_9_stepsUsed_ = out3_
            d_2_steps_ = (d_2_steps_) + (d_9_stepsUsed_)
            generated = d_6_generatedOut_
            if not(d_8_stoppedOnEos_):
                if d_7_stoppedOnOpenSpan_:
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
        with _dafny.label("1"):
            while ((d_2_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("1"):
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
                        d_10_constrainedPrompt_: _dafny.Seq
                        d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_11_next_: _dafny.Seq
                        d_11_next_ = eosToken
                        if not(d_3_seenFrom_):
                            d_12_boostAmount_: _dafny.BigRational
                            d_12_boostAmount_ = _dafny.BigRational('4e0')
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, validTokenGroups, d_12_boostAmount_, eosToken)
                            d_11_next_ = out13_
                        elif True:
                            d_13_wasConstrained_: bool = False
                            out14_: _dafny.Seq
                            out15_: bool
                            out14_, out15_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_11_next_ = out14_
                            d_13_wasConstrained_ = out15_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_11_next_) == (eosToken):
                            raise _dafny.Break("1")
                        elif True:
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                            generated = out16_
                            insideConstrainedOut = out17_
                            currentConstrainedOut = out18_
                            if (not(d_3_seenFrom_)) and ((d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "FROM")))):
                                d_3_seenFrom_ = True
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

