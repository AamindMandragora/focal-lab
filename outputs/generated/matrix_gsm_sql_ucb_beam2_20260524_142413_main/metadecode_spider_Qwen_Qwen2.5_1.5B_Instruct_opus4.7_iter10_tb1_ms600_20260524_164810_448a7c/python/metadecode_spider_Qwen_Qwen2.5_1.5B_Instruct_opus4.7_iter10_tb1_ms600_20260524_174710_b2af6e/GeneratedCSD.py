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
        if (maxSteps) == (0):
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one line in this form: SQL: <<query>>. Do not echo the schema, db_id, db_info, or question. Do not use code fences. Do not generate additional examples. The query must use only the exact table and column names from the schema in the prompt, copy literal values directly from the question, and include every filter the question states.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_unconstrainedAccum_: int
        d_2_unconstrainedAccum_ = 0
        d_3_forceOpenLimit_: int
        d_3_forceOpenLimit_ = 6
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_4_remaining_: int
                    d_4_remaining_ = (maxSteps) - (d_1_steps_)
                    if not(insideConstrainedOut):
                        if (d_2_unconstrainedAccum_) >= (d_3_forceOpenLimit_):
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            generated = out0_
                            insideConstrainedOut = out1_
                            currentConstrainedOut = out2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            cost = d_1_steps_
                        elif True:
                            d_5_chunkBudget_: int
                            if (d_4_remaining_) < (4):
                                d_5_chunkBudget_ = d_4_remaining_
                            elif True:
                                d_5_chunkBudget_ = 4
                            d_6_newGen_: _dafny.Seq
                            d_7_stoppedOnOpen_: bool
                            d_8_stoppedOnEos_: bool
                            d_9_used_: int
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: bool
                            out6_: int
                            out3_, out4_, out5_, out6_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_6_newGen_ = out3_
                            d_7_stoppedOnOpen_ = out4_
                            d_8_stoppedOnEos_ = out5_
                            d_9_used_ = out6_
                            generated = d_6_newGen_
                            d_1_steps_ = (d_1_steps_) + (d_9_used_)
                            d_2_unconstrainedAccum_ = (d_2_unconstrainedAccum_) + (d_9_used_)
                            cost = d_1_steps_
                            if d_8_stoppedOnEos_:
                                raise _dafny.Break("0")
                            if d_7_stoppedOnOpen_:
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                generated = out7_
                                insideConstrainedOut = out8_
                                currentConstrainedOut = out9_
                            elif (d_9_used_) == (0):
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                generated = out10_
                                insideConstrainedOut = out11_
                                currentConstrainedOut = out12_
                                d_1_steps_ = (d_1_steps_) + (1)
                                cost = d_1_steps_
                    elif True:
                        d_10_symBudget_: int
                        if (d_4_remaining_) < (8):
                            d_10_symBudget_ = d_4_remaining_
                        elif True:
                            d_10_symBudget_ = 8
                        d_11_newGen_: _dafny.Seq
                        d_12_newCur_: _dafny.Seq
                        d_13_hitEos_: bool
                        d_14_used_: int
                        out13_: _dafny.Seq
                        out14_: _dafny.Seq
                        out15_: bool
                        out16_: int
                        out13_, out14_, out15_, out16_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, prompt, generated, currentConstrainedOut, d_10_symBudget_, eosToken)
                        d_11_newGen_ = out13_
                        d_12_newCur_ = out14_
                        d_13_hitEos_ = out15_
                        d_14_used_ = out16_
                        generated = d_11_newGen_
                        currentConstrainedOut = d_12_newCur_
                        d_1_steps_ = (d_1_steps_) + (d_14_used_)
                        cost = d_1_steps_
                        if d_13_hitEos_:
                            raise _dafny.Break("0")
                        if (d_14_used_) == (0):
                            d_15_tok_: _dafny.Seq
                            out17_: _dafny.Seq
                            out17_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                            d_15_tok_ = out17_
                            d_1_steps_ = (d_1_steps_) + (1)
                            cost = d_1_steps_
                            raise _dafny.Break("0")
                    pass
            pass
        return generated, insideConstrainedOut, currentConstrainedOut, cost

