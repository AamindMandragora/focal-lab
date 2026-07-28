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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. For EVERY arithmetic calculation, write it inside << >> using the exact format <<expression=result>>. The final answer must also be inside << >>. Example: Janet has 3 apples and gets 5 more. She has <<3+5=8>> apples. Then she eats 2, leaving <<8-2=6>>. The answer is <<6>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_chunkBudget_: int
                        d_2_chunkBudget_ = (maxSteps) - (d_1_steps_)
                        if (d_2_chunkBudget_) > (32):
                            d_2_chunkBudget_ = 32
                        d_3_chunkedG_: _dafny.Seq
                        d_4_stoppedOpen_: bool
                        d_5_stoppedEos_: bool
                        d_6_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_3_chunkedG_ = out0_
                        d_4_stoppedOpen_ = out1_
                        d_5_stoppedEos_ = out2_
                        d_6_stepsUsed_ = out3_
                        generated = d_3_chunkedG_
                        d_1_steps_ = (d_1_steps_) + (d_6_stepsUsed_)
                        if d_5_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_4_stoppedOpen_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif (d_6_stepsUsed_) == (0):
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_7_closedGenerated_: _dafny.Seq
                        d_8_closedInside_: bool
                        d_9_closedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_7_closedGenerated_ = out4_
                        d_8_closedInside_ = out5_
                        d_9_closedCurrent_ = out6_
                        generated = d_7_closedGenerated_
                        insideConstrainedOut = d_8_closedInside_
                        currentConstrainedOut = d_9_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_10_stablePrefix_: _dafny.Seq
                        d_10_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_11_constrainedPrompt_: _dafny.Seq
                        d_11_constrainedPrompt_ = (prompt) + (d_10_stablePrefix_)
                        d_12_remaining_: int
                        d_12_remaining_ = (maxSteps) - (d_1_steps_)
                        d_13_symbolBudget_: int
                        if (d_12_remaining_) < (12):
                            d_13_symbolBudget_ = d_12_remaining_
                        elif True:
                            d_13_symbolBudget_ = 12
                        d_14_prevLen_: int
                        d_14_prevLen_ = len(currentConstrainedOut)
                        d_15_symbolGenerated_: _dafny.Seq
                        d_16_symbolOut_: _dafny.Seq
                        d_17_hitEos_: bool
                        d_18_stepsUsed_: int
                        out7_: _dafny.Seq
                        out8_: _dafny.Seq
                        out9_: bool
                        out10_: int
                        out7_, out8_, out9_, out10_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_11_constrainedPrompt_, generated, currentConstrainedOut, d_13_symbolBudget_, eosToken)
                        d_15_symbolGenerated_ = out7_
                        d_16_symbolOut_ = out8_
                        d_17_hitEos_ = out9_
                        d_18_stepsUsed_ = out10_
                        generated = d_15_symbolGenerated_
                        currentConstrainedOut = d_16_symbolOut_
                        d_1_steps_ = (d_1_steps_) + (d_18_stepsUsed_)
                        if d_17_hitEos_:
                            raise _dafny.Break("0")
                        if (len(currentConstrainedOut)) <= (d_14_prevLen_):
                            if (d_1_steps_) >= (maxSteps):
                                raise _dafny.Break("0")
                            if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                d_19_stablePrefix2_: _dafny.Seq
                                d_19_stablePrefix2_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                                d_20_constrainedPrompt2_: _dafny.Seq
                                d_20_constrainedPrompt2_ = (prompt) + (d_19_stablePrefix2_)
                                d_21_next_: _dafny.Seq
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_20_constrainedPrompt2_, currentConstrainedOut, eosToken)
                                d_21_next_ = out11_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_21_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_22_validNext_: bool
                                    out12_: bool
                                    out12_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_21_next_)
                                    d_22_validNext_ = out12_
                                    if d_22_validNext_:
                                        d_23_appendedGenerated_: _dafny.Seq
                                        d_24_appendedInside_: bool
                                        d_25_appendedCurrent_: _dafny.Seq
                                        out13_: _dafny.Seq
                                        out14_: bool
                                        out15_: _dafny.Seq
                                        out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_21_next_)
                                        d_23_appendedGenerated_ = out13_
                                        d_24_appendedInside_ = out14_
                                        d_25_appendedCurrent_ = out15_
                                        generated = d_23_appendedGenerated_
                                        insideConstrainedOut = d_24_appendedInside_
                                        currentConstrainedOut = d_25_appendedCurrent_
                                    elif True:
                                        raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

