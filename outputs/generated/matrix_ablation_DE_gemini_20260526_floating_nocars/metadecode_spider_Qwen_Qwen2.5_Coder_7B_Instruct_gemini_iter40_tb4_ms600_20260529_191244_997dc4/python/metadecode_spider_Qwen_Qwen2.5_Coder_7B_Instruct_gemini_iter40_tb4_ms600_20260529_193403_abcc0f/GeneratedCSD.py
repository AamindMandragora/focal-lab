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
                        d_8_constrainedPrompt_: _dafny.Seq
                        d_8_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_9_i_: int
                        d_9_i_ = 0
                        while (d_9_i_) < (len(validTokenGroups)):
                            d_10_group_: _dafny.Seq
                            d_10_group_ = (validTokenGroups)[d_9_i_]
                            d_11_tokensInVocab_: _dafny.Seq
                            d_11_tokensInVocab_ = _dafny.SeqWithoutIsStrInference([])
                            d_12_j_: int
                            d_12_j_ = 0
                            while (d_12_j_) < (len(d_10_group_)):
                                d_13_token_: _dafny.Seq
                                d_13_token_ = (d_10_group_)[d_12_j_]
                                if (d_13_token_) in ((lm).Tokens):
                                    d_11_tokensInVocab_ = (d_11_tokensInVocab_) + (_dafny.SeqWithoutIsStrInference([d_13_token_]))
                                d_12_j_ = (d_12_j_) + (1)
                            if (len(d_11_tokensInVocab_)) > (0):
                                (d_0_helpers_).BoostTokenLogits(lm, d_11_tokensInVocab_, _dafny.BigRational('3e0'))
                            d_9_i_ = (d_9_i_) + (1)
                        d_14_next_: _dafny.Seq
                        d_15_wasConstrained_: bool
                        out13_: _dafny.Seq
                        out14_: bool
                        out13_, out14_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_14_next_ = out13_
                        d_15_wasConstrained_ = out14_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_14_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                            generated = out15_
                            insideConstrainedOut = out16_
                            currentConstrainedOut = out17_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

