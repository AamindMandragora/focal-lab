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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Format your final answer exactly as: SQL: <<your SQL query>>")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_2_remaining_: int
                    d_2_remaining_ = (maxSteps) - (d_1_steps_)
                    if not(insideConstrainedOut):
                        d_3_chunkBudget_: int
                        if (d_2_remaining_) < (6):
                            d_3_chunkBudget_ = d_2_remaining_
                        elif True:
                            d_3_chunkBudget_ = 6
                        d_4_newGen_: _dafny.Seq
                        d_5_stoppedOnOpen_: bool
                        d_6_stoppedOnEos_: bool
                        d_7_used_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_4_newGen_ = out0_
                        d_5_stoppedOnOpen_ = out1_
                        d_6_stoppedOnEos_ = out2_
                        d_7_used_ = out3_
                        generated = d_4_newGen_
                        d_1_steps_ = (d_1_steps_) + (d_7_used_)
                        cost = d_1_steps_
                        if d_5_stoppedOnOpen_:
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            generated = out4_
                            insideConstrainedOut = out5_
                            currentConstrainedOut = out6_
                        elif d_6_stoppedOnEos_:
                            raise _dafny.Break("0")
                        elif True:
                            if (d_1_steps_) < (maxSteps):
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                generated = out7_
                                insideConstrainedOut = out8_
                                currentConstrainedOut = out9_
                                d_1_steps_ = (d_1_steps_) + (1)
                                cost = d_1_steps_
                            elif True:
                                raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_8_closedG_: _dafny.Seq
                        d_9_closedI_: bool
                        d_10_closedC_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_8_closedG_ = out10_
                        d_9_closedI_ = out11_
                        d_10_closedC_ = out12_
                        generated = d_8_closedG_
                        insideConstrainedOut = d_9_closedI_
                        currentConstrainedOut = d_10_closedC_
                        d_1_steps_ = (d_1_steps_) + (1)
                        cost = d_1_steps_
                        raise _dafny.Break("0")
                    elif True:
                        d_11_symBudget_: int
                        if (d_2_remaining_) < (8):
                            d_11_symBudget_ = d_2_remaining_
                        elif True:
                            d_11_symBudget_ = 8
                        d_12_newGen_: _dafny.Seq
                        d_13_newCur_: _dafny.Seq
                        d_14_hitEos_: bool
                        d_15_used_: int
                        out13_: _dafny.Seq
                        out14_: _dafny.Seq
                        out15_: bool
                        out16_: int
                        out13_, out14_, out15_, out16_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, prompt, generated, currentConstrainedOut, d_11_symBudget_, eosToken)
                        d_12_newGen_ = out13_
                        d_13_newCur_ = out14_
                        d_14_hitEos_ = out15_
                        d_15_used_ = out16_
                        generated = d_12_newGen_
                        currentConstrainedOut = d_13_newCur_
                        d_1_steps_ = (d_1_steps_) + (d_15_used_)
                        cost = d_1_steps_
                        if d_14_hitEos_:
                            raise _dafny.Break("0")
                        if (d_15_used_) == (0):
                            d_16_tok_: _dafny.Seq
                            out17_: _dafny.Seq
                            out17_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                            d_16_tok_ = out17_
                            d_1_steps_ = (d_1_steps_) + (1)
                            cost = d_1_steps_
                            raise _dafny.Break("0")
                    pass
            pass
        return generated, insideConstrainedOut, currentConstrainedOut, cost

