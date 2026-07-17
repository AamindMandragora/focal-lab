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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. Show your reasoning. At the end, write ONLY the final arithmetic expression (no text) inside << >> delimiters.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_maxFreeChunk_: int
        d_2_maxFreeChunk_ = 40
        d_3_freeChunkTokens_: int
        d_3_freeChunkTokens_ = 0
        d_4_maxSpanTokens_: int
        d_4_maxSpanTokens_ = 30
        d_5_spanTokens_: int
        d_5_spanTokens_ = 0
        d_6_spansEmitted_: int
        d_6_spansEmitted_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_7_shouldForceOpen_: bool
                        d_7_shouldForceOpen_ = (d_3_freeChunkTokens_) >= (d_2_maxFreeChunk_)
                        d_8_nearEnd_: bool
                        d_8_nearEnd_ = ((d_1_steps_) + (10)) >= (maxSteps)
                        if (d_7_shouldForceOpen_) or (d_8_nearEnd_):
                            if (d_1_steps_) < (maxSteps):
                                d_9_og_: _dafny.Seq
                                d_10_oi_: bool
                                d_11_oc_: _dafny.Seq
                                out0_: _dafny.Seq
                                out1_: bool
                                out2_: _dafny.Seq
                                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_9_og_ = out0_
                                d_10_oi_ = out1_
                                d_11_oc_ = out2_
                                generated = d_9_og_
                                insideConstrainedOut = d_10_oi_
                                currentConstrainedOut = d_11_oc_
                                d_1_steps_ = (d_1_steps_) + (1)
                                d_3_freeChunkTokens_ = 0
                                d_5_spanTokens_ = 0
                        elif True:
                            d_12_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_12_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_12_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_next_]))
                                if (d_12_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_3_freeChunkTokens_ = 0
                                    d_5_spanTokens_ = 0
                                elif True:
                                    d_3_freeChunkTokens_ = (d_3_freeChunkTokens_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_13_cg_: _dafny.Seq
                        d_14_ci_: bool
                        d_15_cc_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_13_cg_ = out4_
                        d_14_ci_ = out5_
                        d_15_cc_ = out6_
                        generated = d_13_cg_
                        insideConstrainedOut = d_14_ci_
                        currentConstrainedOut = d_15_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_6_spansEmitted_ = (d_6_spansEmitted_) + (1)
                        d_3_freeChunkTokens_ = 0
                        d_5_spanTokens_ = 0
                        if ((d_6_spansEmitted_) >= (1)) and (((d_1_steps_) + (5)) >= (maxSteps)):
                            raise _dafny.Break("0")
                    elif (d_5_spanTokens_) >= (d_4_maxSpanTokens_):
                        d_16_rg_: _dafny.Seq
                        d_17_rc_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: _dafny.Seq
                        out7_, out8_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_16_rg_ = out7_
                        d_17_rc_ = out8_
                        generated = d_16_rg_
                        currentConstrainedOut = d_17_rc_
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            insideConstrainedOut = True
                        elif True:
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_5_spanTokens_ = 0
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_next_: _dafny.Seq
                        out9_: _dafny.Seq
                        out9_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_19_next_ = out9_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_19_next_) == (eosToken):
                            d_20_rg2_: _dafny.Seq
                            d_21_rc2_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: _dafny.Seq
                            out10_, out11_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_20_rg2_ = out10_
                            d_21_rc2_ = out11_
                            generated = d_20_rg2_
                            currentConstrainedOut = d_21_rc2_
                            if (parser).IsCompletePrefix(currentConstrainedOut):
                                insideConstrainedOut = True
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                        elif True:
                            d_22_ag_: _dafny.Seq
                            d_23_ai_: bool
                            d_24_ac_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: bool
                            out14_: _dafny.Seq
                            out12_, out13_, out14_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                            d_22_ag_ = out12_
                            d_23_ai_ = out13_
                            d_24_ac_ = out14_
                            generated = d_22_ag_
                            insideConstrainedOut = d_23_ai_
                            currentConstrainedOut = d_24_ac_
                            d_5_spanTokens_ = (d_5_spanTokens_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

