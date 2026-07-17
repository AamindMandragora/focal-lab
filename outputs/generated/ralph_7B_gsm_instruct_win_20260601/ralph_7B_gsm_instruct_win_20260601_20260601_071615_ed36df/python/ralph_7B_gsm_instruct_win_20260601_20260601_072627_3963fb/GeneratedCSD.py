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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. For EVERY intermediate arithmetic expression, write it inside << >> delimiters, like <<3+4=7>>. The final answer must also be inside << >> delimiters after ####, like #### <<7>>. Always close every << with a matching >>. Keep expressions concise: no repeated characters, no infinite loops."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_seenHash_: bool
        d_3_seenHash_ = False
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_remaining_: int
                        d_4_remaining_ = (maxSteps) - (d_2_steps_)
                        if ((not(d_3_seenHash_)) and ((d_4_remaining_) <= (20))) and ((d_4_remaining_) >= (4)):
                            d_5_hashNext_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_5_hashNext_ = out0_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_5_hashNext_) == (eosToken):
                                raise _dafny.Break("0")
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_hashNext_]))
                            if (d_5_hashNext_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "####"))):
                                d_3_seenHash_ = True
                            if (d_2_steps_) < (maxSteps):
                                d_6_openedGenerated_: _dafny.Seq
                                d_7_openedInside_: bool
                                d_8_openedCurrent_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_6_openedGenerated_ = out1_
                                d_7_openedInside_ = out2_
                                d_8_openedCurrent_ = out3_
                                generated = d_6_openedGenerated_
                                insideConstrainedOut = d_7_openedInside_
                                currentConstrainedOut = d_8_openedCurrent_
                                d_2_steps_ = (d_2_steps_) + (1)
                        elif True:
                            d_9_chunkMax_: int
                            d_9_chunkMax_ = d_4_remaining_
                            if (d_9_chunkMax_) > (30):
                                d_9_chunkMax_ = 30
                            if (d_9_chunkMax_) == (0):
                                raise _dafny.Break("0")
                            d_10_chunkGenerated_: _dafny.Seq
                            d_11_stoppedOnOpen_: bool
                            d_12_stoppedOnEos_: bool
                            d_13_chunkSteps_: int
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: bool
                            out7_: int
                            out4_, out5_, out6_, out7_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_9_chunkMax_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_10_chunkGenerated_ = out4_
                            d_11_stoppedOnOpen_ = out5_
                            d_12_stoppedOnEos_ = out6_
                            d_13_chunkSteps_ = out7_
                            d_2_steps_ = (d_2_steps_) + (d_13_chunkSteps_)
                            generated = d_10_chunkGenerated_
                            if d_12_stoppedOnEos_:
                                raise _dafny.Break("0")
                            elif d_11_stoppedOnOpen_:
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
                                generated = d_14_enteredGenerated_
                                insideConstrainedOut = d_15_enteredInside_
                                currentConstrainedOut = d_16_enteredCurrent_
                            elif True:
                                d_17_genStr_: _dafny.Seq
                                d_17_genStr_ = VerifiedDecoderAgent.CSDHelpers.PrefixToString(generated)
                                d_18_hashCount_: int
                                d_18_hashCount_ = VerifiedDecoderAgent.CSDHelpers.CountSubstring(d_17_genStr_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "####")))
                                if (d_18_hashCount_) > (0):
                                    d_3_seenHash_ = True
                                if (d_2_steps_) < (maxSteps):
                                    d_19_openedGenerated_: _dafny.Seq
                                    d_20_openedInside_: bool
                                    d_21_openedCurrent_: _dafny.Seq
                                    out11_: _dafny.Seq
                                    out12_: bool
                                    out13_: _dafny.Seq
                                    out11_, out12_, out13_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_19_openedGenerated_ = out11_
                                    d_20_openedInside_ = out12_
                                    d_21_openedCurrent_ = out13_
                                    generated = d_19_openedGenerated_
                                    insideConstrainedOut = d_20_openedInside_
                                    currentConstrainedOut = d_21_openedCurrent_
                                    d_2_steps_ = (d_2_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_22_closedGenerated_: _dafny.Seq
                        d_23_closedInside_: bool
                        d_24_closedCurrent_: _dafny.Seq
                        out14_: _dafny.Seq
                        out15_: bool
                        out16_: _dafny.Seq
                        out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_22_closedGenerated_ = out14_
                        d_23_closedInside_ = out15_
                        d_24_closedCurrent_ = out16_
                        generated = d_22_closedGenerated_
                        insideConstrainedOut = d_23_closedInside_
                        currentConstrainedOut = d_24_closedCurrent_
                        d_2_steps_ = (d_2_steps_) + (1)
                    elif True:
                        d_25_constrainedPrompt_: _dafny.Seq
                        d_25_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_26_isNarrow_: bool
                        out17_: bool
                        out17_ = (d_0_helpers_).DeadEndDetection(parser, currentConstrainedOut, 1)
                        d_26_isNarrow_ = out17_
                        if d_26_isNarrow_:
                            d_27_rolledGenerated_: _dafny.Seq
                            d_28_rolledCurrent_: _dafny.Seq
                            out18_: _dafny.Seq
                            out19_: _dafny.Seq
                            out18_, out19_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_27_rolledGenerated_ = out18_
                            d_28_rolledCurrent_ = out19_
                            generated = d_27_rolledGenerated_
                            currentConstrainedOut = d_28_rolledCurrent_
                            if (d_2_steps_) < (maxSteps):
                                d_29_constrainedPrompt2_: _dafny.Seq
                                d_29_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_30_next_: _dafny.Seq
                                out20_: _dafny.Seq
                                out20_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_29_constrainedPrompt2_, currentConstrainedOut, eosToken)
                                d_30_next_ = out20_
                                d_2_steps_ = (d_2_steps_) + (1)
                                if (d_30_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_31_appendedGenerated_: _dafny.Seq
                                    d_32_appendedInside_: bool
                                    d_33_appendedCurrent_: _dafny.Seq
                                    out21_: _dafny.Seq
                                    out22_: bool
                                    out23_: _dafny.Seq
                                    out21_, out22_, out23_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_30_next_)
                                    d_31_appendedGenerated_ = out21_
                                    d_32_appendedInside_ = out22_
                                    d_33_appendedCurrent_ = out23_
                                    generated = d_31_appendedGenerated_
                                    insideConstrainedOut = d_32_appendedInside_
                                    currentConstrainedOut = d_33_appendedCurrent_
                        elif True:
                            d_34_next_: _dafny.Seq
                            out24_: _dafny.Seq
                            out24_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_25_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('3e0'), eosToken)
                            d_34_next_ = out24_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_34_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_35_appendedGenerated_: _dafny.Seq
                                d_36_appendedInside_: bool
                                d_37_appendedCurrent_: _dafny.Seq
                                out25_: _dafny.Seq
                                out26_: bool
                                out27_: _dafny.Seq
                                out25_, out26_, out27_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_34_next_)
                                d_35_appendedGenerated_ = out25_
                                d_36_appendedInside_ = out26_
                                d_37_appendedCurrent_ = out27_
                                generated = d_35_appendedGenerated_
                                insideConstrainedOut = d_36_appendedInside_
                                currentConstrainedOut = d_37_appendedCurrent_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

