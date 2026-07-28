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
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 20
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_warmup__steps__taken_: int
                        d_3_warmup__steps__taken_ = (len(generated)) - (len(generatedPrefix))
                        if (d_3_warmup__steps__taken_) < (3):
                            d_4_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_4_next_ = out0_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_4_next_) == (eosToken):
                                raise _dafny.Break("0")
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_5_enteredGenerated_: _dafny.Seq
                                d_6_enteredInside_: bool
                                d_7_enteredCurrent_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_5_enteredGenerated_ = out1_
                                d_6_enteredInside_ = out2_
                                d_7_enteredCurrent_ = out3_
                                insideConstrainedOut = d_6_enteredInside_
                                currentConstrainedOut = d_7_enteredCurrent_
                        elif True:
                            d_8_chunkBudget_: int
                            if (maxSteps) > (d_1_steps_):
                                d_8_chunkBudget_ = (maxSteps) - (d_1_steps_)
                            elif True:
                                d_8_chunkBudget_ = 0
                            d_9_chunkedG_: _dafny.Seq
                            d_10_stoppedOpen_: bool
                            d_11_stoppedEos_: bool
                            d_12_stepsUsed_: int
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: bool
                            out7_: int
                            out4_, out5_, out6_, out7_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_8_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_9_chunkedG_ = out4_
                            d_10_stoppedOpen_ = out5_
                            d_11_stoppedEos_ = out6_
                            d_12_stepsUsed_ = out7_
                            generated = d_9_chunkedG_
                            d_1_steps_ = (d_1_steps_) + (d_12_stepsUsed_)
                            if d_11_stoppedEos_:
                                raise _dafny.Break("0")
                            elif d_10_stoppedOpen_:
                                d_13_enteredGenerated_: _dafny.Seq
                                d_14_enteredInside_: bool
                                d_15_enteredCurrent_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_13_enteredGenerated_ = out8_
                                d_14_enteredInside_ = out9_
                                d_15_enteredCurrent_ = out10_
                                insideConstrainedOut = d_14_enteredInside_
                                currentConstrainedOut = d_15_enteredCurrent_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_16_closedGenerated_: _dafny.Seq
                        d_17_closedInside_: bool
                        d_18_closedCurrent_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: _dafny.Seq
                        out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_16_closedGenerated_ = out11_
                        d_17_closedInside_ = out12_
                        d_18_closedCurrent_ = out13_
                        generated = d_16_closedGenerated_
                        insideConstrainedOut = d_17_closedInside_
                        currentConstrainedOut = d_18_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_19_constrainedPrompt_: _dafny.Seq
                        d_19_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_20_validCount_: int
                        out14_: int
                        out14_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_20_validCount_ = out14_
                        if (d_20_validCount_) <= (d_2_narrowThreshold_):
                            d_21_next_: _dafny.Seq
                            out15_: _dafny.Seq
                            out15_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                            d_21_next_ = out15_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_21_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_22_appendedGenerated_: _dafny.Seq
                                d_23_appendedInside_: bool
                                d_24_appendedCurrent_: _dafny.Seq
                                out16_: _dafny.Seq
                                out17_: bool
                                out18_: _dafny.Seq
                                out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                d_22_appendedGenerated_ = out16_
                                d_23_appendedInside_ = out17_
                                d_24_appendedCurrent_ = out18_
                                generated = d_22_appendedGenerated_
                                insideConstrainedOut = d_23_appendedInside_
                                currentConstrainedOut = d_24_appendedCurrent_
                        elif True:
                            d_25_stablePrefix_: _dafny.Seq
                            d_25_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_26_symbolBudget_: int
                            d_26_symbolBudget_ = (maxSteps) - (d_1_steps_)
                            d_27_symbolGenerated_: _dafny.Seq
                            d_28_symbolOut_: _dafny.Seq
                            d_29_hitEos_: bool
                            d_30_stepsUsed_: int
                            out19_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: bool
                            out22_: int
                            out19_, out20_, out21_, out22_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_19_constrainedPrompt_, generated, currentConstrainedOut, d_26_symbolBudget_, eosToken)
                            d_27_symbolGenerated_ = out19_
                            d_28_symbolOut_ = out20_
                            d_29_hitEos_ = out21_
                            d_30_stepsUsed_ = out22_
                            generated = d_27_symbolGenerated_
                            currentConstrainedOut = d_28_symbolOut_
                            d_1_steps_ = (d_1_steps_) + (d_30_stepsUsed_)
                            if d_29_hitEos_:
                                raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

