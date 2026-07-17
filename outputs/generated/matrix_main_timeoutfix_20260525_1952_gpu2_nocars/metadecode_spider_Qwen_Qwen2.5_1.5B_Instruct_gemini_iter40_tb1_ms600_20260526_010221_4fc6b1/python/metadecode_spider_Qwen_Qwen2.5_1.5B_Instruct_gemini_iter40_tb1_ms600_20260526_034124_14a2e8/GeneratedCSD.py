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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Produce a single SQL query in the format: SQL: <<query>>")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_warmup__budget_: int
        d_2_warmup__budget_ = 15
        d_3_warmup__done_: bool
        d_3_warmup__done_ = False
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(d_3_warmup__done_):
                        d_4_budget__for__warmup_: int
                        if ((maxSteps) - (d_1_steps_)) < (d_2_warmup__budget_):
                            d_4_budget__for__warmup_ = (maxSteps) - (d_1_steps_)
                        elif True:
                            d_4_budget__for__warmup_ = d_2_warmup__budget_
                        if (d_4_budget__for__warmup_) > (0):
                            d_5_warmup__gen_: _dafny.Seq
                            d_6___v0_: bool
                            d_7_stoppedEos__warmup_: bool
                            d_8_steps__warmup_: int
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: bool
                            out3_: int
                            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_budget__for__warmup_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_5_warmup__gen_ = out0_
                            d_6___v0_ = out1_
                            d_7_stoppedEos__warmup_ = out2_
                            d_8_steps__warmup_ = out3_
                            generated = d_5_warmup__gen_
                            d_1_steps_ = (d_1_steps_) + (d_8_steps__warmup_)
                            if d_7_stoppedEos__warmup_:
                                raise _dafny.Break("0")
                        d_3_warmup__done_ = True
                    elif not(insideConstrainedOut):
                        d_9_chunkBudget_: int
                        d_9_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        d_10_chunkedG_: _dafny.Seq
                        d_11_stoppedOpen_: bool
                        d_12_stoppedEos_: bool
                        d_13_stepsUsed_: int
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: bool
                        out7_: int
                        out4_, out5_, out6_, out7_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_9_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_10_chunkedG_ = out4_
                        d_11_stoppedOpen_ = out5_
                        d_12_stoppedEos_ = out6_
                        d_13_stepsUsed_ = out7_
                        generated = d_10_chunkedG_
                        d_1_steps_ = (d_1_steps_) + (d_13_stepsUsed_)
                        if d_12_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_11_stoppedOpen_:
                            d_14_enteredGenerated_: _dafny.Seq
                            d_15_enteredInside_: bool
                            d_16_enteredCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_14_enteredGenerated_ = out8_
                            d_15_enteredInside_ = out9_
                            d_16_enteredCurrent_ = out10_
                            insideConstrainedOut = d_15_enteredInside_
                            currentConstrainedOut = d_16_enteredCurrent_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_17_closedGenerated_: _dafny.Seq
                        d_18_closedInside_: bool
                        d_19_closedCurrent_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: _dafny.Seq
                        out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_17_closedGenerated_ = out11_
                        d_18_closedInside_ = out12_
                        d_19_closedCurrent_ = out13_
                        generated = d_17_closedGenerated_
                        insideConstrainedOut = d_18_closedInside_
                        currentConstrainedOut = d_19_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_20_stablePrefix_: _dafny.Seq
                        d_20_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_21_constrainedPrompt_: _dafny.Seq
                        d_21_constrainedPrompt_ = (prompt) + (d_20_stablePrefix_)
                        d_22_symbolBudget_: int
                        d_22_symbolBudget_ = (maxSteps) - (d_1_steps_)
                        d_23_symbolGenerated_: _dafny.Seq
                        d_24_symbolOut_: _dafny.Seq
                        d_25_hitEos_: bool
                        d_26_stepsUsed_: int
                        out14_: _dafny.Seq
                        out15_: _dafny.Seq
                        out16_: bool
                        out17_: int
                        out14_, out15_, out16_, out17_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_21_constrainedPrompt_, generated, currentConstrainedOut, d_22_symbolBudget_, eosToken)
                        d_23_symbolGenerated_ = out14_
                        d_24_symbolOut_ = out15_
                        d_25_hitEos_ = out16_
                        d_26_stepsUsed_ = out17_
                        generated = d_23_symbolGenerated_
                        currentConstrainedOut = d_24_symbolOut_
                        d_1_steps_ = (d_1_steps_) + (d_26_stepsUsed_)
                        if d_25_hitEos_:
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

