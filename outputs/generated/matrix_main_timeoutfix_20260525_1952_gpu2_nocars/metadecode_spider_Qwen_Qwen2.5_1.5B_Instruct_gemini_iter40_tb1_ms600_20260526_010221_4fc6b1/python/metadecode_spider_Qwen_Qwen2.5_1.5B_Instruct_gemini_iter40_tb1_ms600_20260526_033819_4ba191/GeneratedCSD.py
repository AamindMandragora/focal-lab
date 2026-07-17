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
        d_2_mandatory__warmup__tokens_: int
        d_2_mandatory__warmup__tokens_ = 10
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_1_steps_) < (d_2_mandatory__warmup__tokens_):
                            d_3_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_3_next_ = out0_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_3_next_) == (eosToken):
                                raise _dafny.Break("0")
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                        elif True:
                            d_4_chunkBudget_: int
                            d_4_chunkBudget_ = (maxSteps) - (d_1_steps_)
                            d_5_chunkedG_: _dafny.Seq
                            d_6_stoppedOpen_: bool
                            d_7_stoppedEos_: bool
                            d_8_stepsUsed_: int
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: bool
                            out4_: int
                            out1_, out2_, out3_, out4_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_4_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_5_chunkedG_ = out1_
                            d_6_stoppedOpen_ = out2_
                            d_7_stoppedEos_ = out3_
                            d_8_stepsUsed_ = out4_
                            generated = d_5_chunkedG_
                            d_1_steps_ = (d_1_steps_) + (d_8_stepsUsed_)
                            if d_7_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_6_stoppedOpen_:
                                d_9_enteredGenerated_: _dafny.Seq
                                d_10_enteredInside_: bool
                                d_11_enteredCurrent_: _dafny.Seq
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: _dafny.Seq
                                out5_, out6_, out7_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_9_enteredGenerated_ = out5_
                                d_10_enteredInside_ = out6_
                                d_11_enteredCurrent_ = out7_
                                insideConstrainedOut = d_10_enteredInside_
                                currentConstrainedOut = d_11_enteredCurrent_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_12_closedGenerated_: _dafny.Seq
                        d_13_closedInside_: bool
                        d_14_closedCurrent_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: _dafny.Seq
                        out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_12_closedGenerated_ = out8_
                        d_13_closedInside_ = out9_
                        d_14_closedCurrent_ = out10_
                        generated = d_12_closedGenerated_
                        insideConstrainedOut = d_13_closedInside_
                        currentConstrainedOut = d_14_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_15_stablePrefix_: _dafny.Seq
                        d_15_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_16_constrainedPrompt_: _dafny.Seq
                        d_16_constrainedPrompt_ = (prompt) + (d_15_stablePrefix_)
                        d_17_symbolBudget_: int
                        d_17_symbolBudget_ = (maxSteps) - (d_1_steps_)
                        d_18_symbolGenerated_: _dafny.Seq
                        d_19_symbolOut_: _dafny.Seq
                        d_20_hitEos_: bool
                        d_21_stepsUsed_: int
                        out11_: _dafny.Seq
                        out12_: _dafny.Seq
                        out13_: bool
                        out14_: int
                        out11_, out12_, out13_, out14_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_16_constrainedPrompt_, generated, currentConstrainedOut, d_17_symbolBudget_, eosToken)
                        d_18_symbolGenerated_ = out11_
                        d_19_symbolOut_ = out12_
                        d_20_hitEos_ = out13_
                        d_21_stepsUsed_ = out14_
                        generated = d_18_symbolGenerated_
                        currentConstrainedOut = d_19_symbolOut_
                        d_1_steps_ = (d_1_steps_) + (d_21_stepsUsed_)
                        if d_20_hitEos_:
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

