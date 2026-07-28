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
        (d_0_helpers_).AppendTaskGuidance(lm, (((((((_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output format rules (strict):\n"))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "- Solve in 2-4 short steps.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "- Keep the original variable names exactly as written (e.g. n_1 stays n_1, frac_2 stays frac_2). Do not invent new variables.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "- Every arithmetic computation must be wrapped in << >> in the form <<expression=result>>, where expression is a Python expression over the given variables.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "- Use Python integer arithmetic. Use // for division when the result must be a whole number. Wrap any expression whose mathematical value may not be an integer in int(...), e.g. int(a*b/c).\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "- Do NOT substitute numeric values for the variables; keep the answer symbolic.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "- The very last line must be exactly: #### <<FINAL_EXPRESSION>> where FINAL_EXPRESSION is the symbolic answer (using // and int(...) as needed). Stop immediately after that line.\n")))) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Worked example. Problem variables: a, b, c. Answer: <<a*b=ab>>. <<ab//c=r>>. #### <<a*b//c>>"))))
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
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_3_closedGenerated_: _dafny.Seq
                        d_4_closedInside_: bool
                        d_5_closedCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_3_closedGenerated_ = out1_
                        d_4_closedInside_ = out2_
                        d_5_closedCurrent_ = out3_
                        generated = d_3_closedGenerated_
                        insideConstrainedOut = d_4_closedInside_
                        currentConstrainedOut = d_5_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_6_constrainedPrompt_: _dafny.Seq
                        d_6_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_7_next_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_6_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_7_next_ = out4_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_7_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_8_appendedGenerated_: _dafny.Seq
                            d_9_appendedInside_: bool
                            d_10_appendedCurrent_: _dafny.Seq
                            out5_: _dafny.Seq
                            out6_: bool
                            out7_: _dafny.Seq
                            out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_7_next_)
                            d_8_appendedGenerated_ = out5_
                            d_9_appendedInside_ = out6_
                            d_10_appendedCurrent_ = out7_
                            generated = d_8_appendedGenerated_
                            insideConstrainedOut = d_9_appendedInside_
                            currentConstrainedOut = d_10_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

