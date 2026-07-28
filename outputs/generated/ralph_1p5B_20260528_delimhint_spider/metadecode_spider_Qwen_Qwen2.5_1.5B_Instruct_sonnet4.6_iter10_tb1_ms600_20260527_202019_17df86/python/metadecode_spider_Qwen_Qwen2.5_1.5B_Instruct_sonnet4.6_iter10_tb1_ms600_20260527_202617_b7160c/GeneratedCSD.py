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
        (d_0_helpers_).AppendTaskGuidance(lm, (((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You must answer with exactly one SQL query in this format: SQL: <<SELECT ...>>. "))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Use only the table and column names from the provided schema. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Include WHERE, GROUP BY, HAVING, ORDER BY, LIMIT clauses as needed. ")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Do not add explanations or extra text."))))
        d_1_steps_: int
        d_1_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_2_preambleBudget_: int
            if ((maxSteps) - (d_1_steps_)) >= (8):
                d_2_preambleBudget_ = 8
            elif True:
                d_2_preambleBudget_ = (maxSteps) - (d_1_steps_)
            d_3_chunkOut_: _dafny.Seq
            d_4_stoppedOpen_: bool
            d_5_stoppedEos_: bool
            d_6_chunkUsed_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_preambleBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_3_chunkOut_ = out0_
            d_4_stoppedOpen_ = out1_
            d_5_stoppedEos_ = out2_
            d_6_chunkUsed_ = out3_
            generated = d_3_chunkOut_
            d_1_steps_ = (d_1_steps_) + (d_6_chunkUsed_)
            if d_5_stoppedEos_:
                pass
            elif d_4_stoppedOpen_:
                d_7_gOut_: _dafny.Seq
                d_8_iOut_: bool
                d_9_cOut_: _dafny.Seq
                out4_: _dafny.Seq
                out5_: bool
                out6_: _dafny.Seq
                out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                d_7_gOut_ = out4_
                d_8_iOut_ = out5_
                d_9_cOut_ = out6_
                generated = d_7_gOut_
                insideConstrainedOut = d_8_iOut_
                currentConstrainedOut = d_9_cOut_
            elif (d_1_steps_) < (maxSteps):
                d_10_gOut_: _dafny.Seq
                d_11_iOut_: bool
                d_12_cOut_: _dafny.Seq
                out7_: _dafny.Seq
                out8_: bool
                out9_: _dafny.Seq
                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_10_gOut_ = out7_
                d_11_iOut_ = out8_
                d_12_cOut_ = out9_
                generated = d_10_gOut_
                insideConstrainedOut = d_11_iOut_
                currentConstrainedOut = d_12_cOut_
                d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_13_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_13_next_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_13_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_13_next_]))
                            if (d_13_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_14_gOut_: _dafny.Seq
                                d_15_iOut_: bool
                                d_16_cOut_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_14_gOut_ = out11_
                                d_15_iOut_ = out12_
                                d_16_cOut_ = out13_
                                generated = d_14_gOut_
                                insideConstrainedOut = d_15_iOut_
                                currentConstrainedOut = d_16_cOut_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_17_gOut_: _dafny.Seq
                        d_18_iOut_: bool
                        d_19_cOut_: _dafny.Seq
                        out14_: _dafny.Seq
                        out15_: bool
                        out16_: _dafny.Seq
                        out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_17_gOut_ = out14_
                        d_18_iOut_ = out15_
                        d_19_cOut_ = out16_
                        generated = d_17_gOut_
                        insideConstrainedOut = d_18_iOut_
                        currentConstrainedOut = d_19_cOut_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_20_constrainedPrompt_: _dafny.Seq
                        d_20_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_21_next_: _dafny.Seq
                        d_22_wasConstrained_: bool
                        out17_: _dafny.Seq
                        out18_: bool
                        out17_, out18_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_20_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_21_next_ = out17_
                        d_22_wasConstrained_ = out18_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_21_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_23_gOut_: _dafny.Seq
                            d_24_iOut_: bool
                            d_25_cOut_: _dafny.Seq
                            out19_: _dafny.Seq
                            out20_: bool
                            out21_: _dafny.Seq
                            out19_, out20_, out21_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                            d_23_gOut_ = out19_
                            d_24_iOut_ = out20_
                            d_25_cOut_ = out21_
                            generated = d_23_gOut_
                            insideConstrainedOut = d_24_iOut_
                            currentConstrainedOut = d_25_cOut_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

