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
        (d_0_helpers_).AppendTaskGuidance(lm, (((((((((((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Symbolic math word problem. Variables in braces like {n}, {p}, {tax} are placeholder names. In your answer write them WITHOUT braces (write n, not {n}). Solve briefly step by step in plain English, then put the FINAL complete formula combining ALL needed variables inside << and >>. The grader reads ONLY the LAST << >> span. Use only variable names and the operators + - * / ( ). Stop immediately after the closing >>. Do NOT write multiple << >> spans for intermediate steps; only the final formula should be in << >>.\n\n"))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Example 1:\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Q: A store has {n} items priced at {p} each. Sales tax is {tax}%. What is the total cost?\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "A: Subtotal is n*p, plus tax gives n*p*(1+tax/100). <<n*p*(1+tax/100)>>\n\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Example 2:\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Q: {name} has {n1} marbles, gives {n2} to a friend, then finds {n3} more. How many marbles does {name} have?\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "A: Start n1, subtract n2, add n3. <<n1-n2+n3>>\n\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Example 3:\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Q: A bag has {cal} calories per serving and {n} servings. If daily target is {total} and {spent} already consumed, how many bags can you eat?\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "A: Per bag = cal*n calories. Remaining = total-spent. Bags = (total-spent)/(cal*n). <<(total-spent)/(cal*n)>>\n\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Example 4:\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Q: {a} dog has {n1} puppies, {k1} spotted. {b} dog has {n2} puppies, {k2} spotted. What percent of all puppies are spotted?\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "A: Total puppies = n1+n2. Total spotted = k1+k2. Percent = 100*(k1+k2)/(n1+n2). <<100*(k1+k2)/(n1+n2)>>\n\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Now solve the question below. Reason briefly, then end with the complete formula in << >>."))))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_chunkBudget_: int
                        d_2_chunkBudget_ = (maxSteps) - (d_1_steps_)
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
                        elif True:
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
                        d_10_constrainedPrompt_: _dafny.Seq
                        d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_11_next_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_11_next_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_11_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_12_appendedGenerated_: _dafny.Seq
                            d_13_appendedInside_: bool
                            d_14_appendedCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_next_)
                            d_12_appendedGenerated_ = out8_
                            d_13_appendedInside_ = out9_
                            d_14_appendedCurrent_ = out10_
                            generated = d_12_appendedGenerated_
                            insideConstrainedOut = d_13_appendedInside_
                            currentConstrainedOut = d_14_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

