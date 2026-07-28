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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the problem step by step. Wrap every arithmetic calculation in the GSM format <<expression=result>>. Examples: <<5+3=8>>, <<24/4=6>>, <<100-25=75>>. Inside << >> use only digits, variable names, the operators + - * /, parentheses, and exactly one '=' before the numeric result; no LaTeX, no units, no words. Always close each << with >> before continuing. End the solution with: The answer is <<final_expression=final_value>>.")))
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
                            d_3_openSpan_: bool
                            d_3_openSpan_ = False
                            if (d_2_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_3_openSpan_ = True
                            if (not(d_3_openSpan_)) and ((len(d_2_next_)) >= (2)):
                                d_4_j_: int
                                d_4_j_ = 0
                                with _dafny.label("0_0_1_1_0"):
                                    while ((d_4_j_) + (1)) < (len(d_2_next_)):
                                        with _dafny.c_label("0_0_1_1_0"):
                                            if (((d_2_next_)[d_4_j_]) == (_dafny.CodePoint('<'))) and (((d_2_next_)[(d_4_j_) + (1)]) == (_dafny.CodePoint('<'))):
                                                d_3_openSpan_ = True
                                                raise _dafny.Break("0_0_1_1_0")
                                            d_4_j_ = (d_4_j_) + (1)
                                            pass
                                    pass
                            if (((not(d_3_openSpan_)) and ((len(d_2_next_)) >= (1))) and (((d_2_next_)[0]) == (_dafny.CodePoint('<')))) and ((len(generated)) >= (2)):
                                d_5_prev_: _dafny.Seq
                                d_5_prev_ = (generated)[(len(generated)) - (2)]
                                if ((len(d_5_prev_)) >= (1)) and (((d_5_prev_)[(len(d_5_prev_)) - (1)]) == (_dafny.CodePoint('<'))):
                                    d_3_openSpan_ = True
                            if d_3_openSpan_:
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_6_closedGenerated_: _dafny.Seq
                        d_7_closedInside_: bool
                        d_8_closedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_6_closedGenerated_ = out1_
                        d_7_closedInside_ = out2_
                        d_8_closedCurrent_ = out3_
                        generated = d_6_closedGenerated_
                        insideConstrainedOut = d_7_closedInside_
                        currentConstrainedOut = d_8_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_9_constrainedPrompt_: _dafny.Seq
                        d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_10_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if (len(currentConstrainedOut)) >= (16):
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('6e0'), eosToken)
                            d_10_next_ = out4_
                        elif True:
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_10_next_ = out5_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_10_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_11_appendedGenerated_: _dafny.Seq
                            d_12_appendedInside_: bool
                            d_13_appendedCurrent_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                            d_11_appendedGenerated_ = out6_
                            d_12_appendedInside_ = out7_
                            d_13_appendedCurrent_ = out8_
                            generated = d_11_appendedGenerated_
                            insideConstrainedOut = d_12_appendedInside_
                            currentConstrainedOut = d_13_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

