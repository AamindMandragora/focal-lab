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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve math word problems step by step. For each arithmetic step, write the symbolic expression inside << >> delimiters. For example: <<3 + 4 = 7>>. End with #### <<answer>>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_2_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_2_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                            if (d_2_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_3_enteredGenerated_: _dafny.Seq
                                d_4_enteredInside_: bool
                                d_5_enteredCurrent_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_3_enteredGenerated_ = out1_
                                d_4_enteredInside_ = out2_
                                d_5_enteredCurrent_ = out3_
                                generated = d_3_enteredGenerated_
                                insideConstrainedOut = d_4_enteredInside_
                                currentConstrainedOut = d_5_enteredCurrent_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        if (d_1_steps_) >= (maxSteps):
                            raise _dafny.Break("0")
                        d_6_closedGenerated_: _dafny.Seq
                        d_7_closedInside_: bool
                        d_8_closedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_6_closedGenerated_ = out4_
                        d_7_closedInside_ = out5_
                        d_8_closedCurrent_ = out6_
                        generated = d_6_closedGenerated_
                        insideConstrainedOut = d_7_closedInside_
                        currentConstrainedOut = d_8_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_9_constrainedPrompt_: _dafny.Seq
                        d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_10_next_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_10_next_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_10_next_) == (eosToken):
                            d_11_rolledGenerated_: _dafny.Seq
                            d_12_rolledCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: _dafny.Seq
                            out8_, out9_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                            d_11_rolledGenerated_ = out8_
                            d_12_rolledCurrent_ = out9_
                            generated = d_11_rolledGenerated_
                            currentConstrainedOut = d_12_rolledCurrent_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_13_closedGenerated_: _dafny.Seq
                                d_14_closedInside_: bool
                                d_15_closedCurrent_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_13_closedGenerated_ = out10_
                                d_14_closedInside_ = out11_
                                d_15_closedCurrent_ = out12_
                                generated = d_13_closedGenerated_
                                insideConstrainedOut = d_14_closedInside_
                                currentConstrainedOut = d_15_closedCurrent_
                                d_1_steps_ = (d_1_steps_) + (1)
                            elif True:
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                raise _dafny.Break("0")
                        elif True:
                            d_16_valid_: bool
                            out13_: bool
                            out13_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_10_next_)
                            d_16_valid_ = out13_
                            if d_16_valid_:
                                d_17_appendedGenerated_: _dafny.Seq
                                d_18_appendedInside_: bool
                                d_19_appendedCurrent_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                                d_17_appendedGenerated_ = out14_
                                d_18_appendedInside_ = out15_
                                d_19_appendedCurrent_ = out16_
                                generated = d_17_appendedGenerated_
                                insideConstrainedOut = d_18_appendedInside_
                                currentConstrainedOut = d_19_appendedCurrent_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_20_closedGenerated_: _dafny.Seq
                                    d_21_closedInside_: bool
                                    d_22_closedCurrent_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out17_, out18_, out19_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_20_closedGenerated_ = out17_
                                    d_21_closedInside_ = out18_
                                    d_22_closedCurrent_ = out19_
                                    generated = d_20_closedGenerated_
                                    insideConstrainedOut = d_21_closedInside_
                                    currentConstrainedOut = d_22_closedCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

