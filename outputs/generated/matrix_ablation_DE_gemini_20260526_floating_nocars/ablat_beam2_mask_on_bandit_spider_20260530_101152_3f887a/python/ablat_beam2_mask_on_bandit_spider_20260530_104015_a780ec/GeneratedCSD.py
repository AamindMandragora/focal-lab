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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one SQL query using only the provided schema context. The final answer surface is SQL: <<YOUR QUERY>>. Continue inside the delimiters with only the SQL query body, no explanation, no Markdown, no comments, and no extra prose. Use the actual schema table and column names from the prompt. Choose joins by foreign-key or matching id columns, prefer semantically correct tables over merely similar names, and close the query with >> immediately after the SQL."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Treat the supplied contextual token groups as schema hints, but use them only when they match the question.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_headerDone_: bool
        d_2_headerDone_ = (insideConstrainedOut) or ((len(generated)) > (0))
        d_3_openCount_: int
        out0_: int
        out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
        d_3_openCount_ = out0_
        d_4_closeCount_: int
        out1_: int
        out1_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
        d_4_closeCount_ = out1_
        d_5_spanStarted_: bool
        d_5_spanStarted_ = (insideConstrainedOut) or ((d_3_openCount_) > (0))
        d_6_spanClosed_: bool
        d_6_spanClosed_ = (d_4_closeCount_) > (0)
        d_7_steps_: int
        d_7_steps_ = 0
        with _dafny.label("0"):
            while (d_7_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_8_closedGenerated_: _dafny.Seq
                            d_9_closedInside_: bool
                            d_10_closedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_8_closedGenerated_ = out2_
                            d_9_closedInside_ = out3_
                            d_10_closedCurrent_ = out4_
                            generated = d_8_closedGenerated_
                            insideConstrainedOut = d_9_closedInside_
                            currentConstrainedOut = d_10_closedCurrent_
                            d_7_steps_ = (d_7_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_11_constrainedPrompt_: _dafny.Seq
                            d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_12_next_: _dafny.Seq
                            d_13_wasConstrained_: bool
                            out5_: _dafny.Seq
                            out6_: bool
                            out5_, out6_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_12_next_ = out5_
                            d_13_wasConstrained_ = out6_
                            d_7_steps_ = (d_7_steps_) + (1)
                            if (d_12_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_14_appendedGenerated_: _dafny.Seq
                                d_15_appendedInside_: bool
                                d_16_appendedCurrent_: _dafny.Seq
                                out7_: _dafny.Seq
                                out8_: bool
                                out9_: _dafny.Seq
                                out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                                d_14_appendedGenerated_ = out7_
                                d_15_appendedInside_ = out8_
                                d_16_appendedCurrent_ = out9_
                                generated = d_14_appendedGenerated_
                                insideConstrainedOut = d_15_appendedInside_
                                currentConstrainedOut = d_16_appendedCurrent_
                    elif True:
                        if not(d_2_headerDone_):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL: "))]))
                            d_2_headerDone_ = True
                            d_7_steps_ = (d_7_steps_) + (1)
                        elif not(d_5_spanStarted_):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
                            d_5_spanStarted_ = True
                            d_7_steps_ = (d_7_steps_) + (1)
                        elif d_6_spanClosed_:
                            d_17_sink_: _dafny.Seq
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_17_sink_ = out10_
                            d_7_steps_ = (d_7_steps_) + (1)
                            raise _dafny.Break("0")
                        elif ((d_7_steps_) + (1)) == (maxSteps):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                            d_6_spanClosed_ = True
                            d_7_steps_ = (d_7_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_18_nextFree_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_18_nextFree_ = out11_
                            d_7_steps_ = (d_7_steps_) + (1)
                            if (d_18_nextFree_) == (eosToken):
                                if (d_7_steps_) < (maxSteps):
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                                    d_6_spanClosed_ = True
                                    d_7_steps_ = (d_7_steps_) + (1)
                                raise _dafny.Break("0")
                            elif (d_18_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_18_nextFree_]))
                                d_6_spanClosed_ = True
                                raise _dafny.Break("0")
                            elif (((d_18_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or ((d_18_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL: "))))) or ((d_18_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:")))):
                                pass
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_18_nextFree_]))
                    pass
            pass
        cost = d_7_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

