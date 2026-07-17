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
        (d_0_helpers_).AppendTaskGuidance(lm, (((((((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OUTPUT FORMAT - read carefully:\n"))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1. Solve the word problem step by step in plain prose.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2. Each intermediate arithmetic step uses <<expr=value>>, e.g. <<3*4=12>>.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3. The VERY LAST << >> in your answer must contain the COMPLETE final answer as a single fully-simplified symbolic expression wrapped in int(...). Examples of correct final spans:\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "   <<int(n1*c1 + n2*c2 + c3)>>\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "   <<int((2*n1+n2)*cn + (2*m2-m1)*cm)>>\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "   <<int(p*q - r)>>\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4. In the final << >>: NO `=` sign, NO whitespace inside, use explicit `*` for every multiplication (write `2*n1`, never `2n1`), combine all like terms, and wrap the whole expression with int( ).\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5. The final << >> must include ALL components of the answer, not just one intermediate piece. Stop immediately after closing the final >>.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6. Do not write >>>> or nest << inside << >>."))))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_maxChunk_: int
        d_2_maxChunk_ = 40
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_remaining_: int
                        d_3_remaining_ = (maxSteps) - (d_1_steps_)
                        d_4_reserve_: int
                        d_4_reserve_ = 12
                        d_5_chunkBudget_: int
                        if (d_3_remaining_) > (d_4_reserve_):
                            d_5_chunkBudget_ = (d_3_remaining_) - (d_4_reserve_)
                        elif True:
                            d_5_chunkBudget_ = d_3_remaining_
                        if (d_5_chunkBudget_) > (d_2_maxChunk_):
                            d_5_chunkBudget_ = d_2_maxChunk_
                        if (d_5_chunkBudget_) == (0):
                            d_5_chunkBudget_ = d_3_remaining_
                        d_6_chunkedG_: _dafny.Seq
                        d_7_stoppedOpen_: bool
                        d_8_stoppedEos_: bool
                        d_9_stepsUsed_: int
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: bool
                        out3_: int
                        out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_5_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                        d_6_chunkedG_ = out0_
                        d_7_stoppedOpen_ = out1_
                        d_8_stoppedEos_ = out2_
                        d_9_stepsUsed_ = out3_
                        generated = d_6_chunkedG_
                        d_1_steps_ = (d_1_steps_) + (d_9_stepsUsed_)
                        if d_8_stoppedEos_:
                            raise _dafny.Break("0")
                        elif d_7_stoppedOpen_:
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        elif (d_9_stepsUsed_) == (0):
                            raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_10_closedGenerated_: _dafny.Seq
                        d_11_closedInside_: bool
                        d_12_closedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_10_closedGenerated_ = out4_
                        d_11_closedInside_ = out5_
                        d_12_closedCurrent_ = out6_
                        generated = d_10_closedGenerated_
                        insideConstrainedOut = d_11_closedInside_
                        currentConstrainedOut = d_12_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_13_constrainedPrompt_: _dafny.Seq
                        d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_14_next_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_14_next_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_14_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_15_appendedGenerated_: _dafny.Seq
                            d_16_appendedInside_: bool
                            d_17_appendedCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                            d_15_appendedGenerated_ = out8_
                            d_16_appendedInside_ = out9_
                            d_17_appendedCurrent_ = out10_
                            generated = d_15_appendedGenerated_
                            insideConstrainedOut = d_16_appendedInside_
                            currentConstrainedOut = d_17_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

