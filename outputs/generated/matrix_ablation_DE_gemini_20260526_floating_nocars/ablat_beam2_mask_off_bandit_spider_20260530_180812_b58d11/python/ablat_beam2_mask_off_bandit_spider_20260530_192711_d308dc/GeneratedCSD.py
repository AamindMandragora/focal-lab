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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one valid SQLite query using only the provided schema context. Output exactly SQL: <<YOUR QUERY>> with no explanation, no Markdown, and no extra text. The decoder will force SQL: << and the closing delimiter; continue only with the SQL query. Use exact table and column names from the schema. Prefer the simplest semantically correct query. If a question asks for records not used or not in another set, consider EXCEPT or NOT IN. If it asks for items satisfying both conditions, consider INTERSECT or grouped filtering. Join tables explicitly when selected or filtered columns come from different tables. Avoid inventing aliases unless they make joins clearer, and avoid semicolons."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Treat contextual token groups as schema identifier hints, but only use them when they match the question and schema.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_headerStage_: int
        d_2_headerStage_ = 0
        d_3_steps_: int
        d_3_steps_ = 0
        with _dafny.label("0"):
            while (d_3_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_2_headerStage_) == (0):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:"))]))
                            d_2_headerStage_ = 1
                            d_3_steps_ = (d_3_steps_) + (1)
                        elif (d_2_headerStage_) == (1):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " "))]))
                            d_2_headerStage_ = 2
                            d_3_steps_ = (d_3_steps_) + (1)
                        elif True:
                            d_4_openedGenerated_: _dafny.Seq
                            d_5_openedInside_: bool
                            d_6_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_openedGenerated_ = out0_
                            d_5_openedInside_ = out1_
                            d_6_openedCurrent_ = out2_
                            generated = d_4_openedGenerated_
                            insideConstrainedOut = d_5_openedInside_
                            currentConstrainedOut = d_6_openedCurrent_
                            d_3_steps_ = (d_3_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_7_constrainedPrompt_: _dafny.Seq
                        d_7_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_8_nextLook_: _dafny.Seq
                        d_9_wasConstrained_: bool
                        out3_: _dafny.Seq
                        out4_: bool
                        out3_, out4_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_8_nextLook_ = out3_
                        d_9_wasConstrained_ = out4_
                        d_3_steps_ = (d_3_steps_) + (1)
                        if (((((d_8_nextLook_) == (eosToken)) or (d_9_wasConstrained_)) or ((d_8_nextLook_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ";"))))) or ((d_8_nextLook_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))))) or ((d_8_nextLook_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))):
                            if (d_3_steps_) < (maxSteps):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_3_steps_ = (d_3_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_10_validLook_: bool
                            out5_: bool
                            out5_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_8_nextLook_)
                            d_10_validLook_ = out5_
                            if d_10_validLook_:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_nextLook_]))
                                currentConstrainedOut = (currentConstrainedOut) + (_dafny.SeqWithoutIsStrInference([d_8_nextLook_]))
                            elif True:
                                if (d_3_steps_) < (maxSteps):
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                                    insideConstrainedOut = False
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_3_steps_ = (d_3_steps_) + (1)
                                raise _dafny.Break("0")
                    elif True:
                        d_11_constrainedPrompt2_: _dafny.Seq
                        d_11_constrainedPrompt2_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_12_nextSql_: _dafny.Seq
                        out6_: _dafny.Seq
                        out6_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_11_constrainedPrompt2_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('8e0'), 12, eosToken)
                        d_12_nextSql_ = out6_
                        d_3_steps_ = (d_3_steps_) + (1)
                        if (d_12_nextSql_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_13_appendedGenerated_: _dafny.Seq
                            d_14_appendedInside_: bool
                            d_15_appendedCurrent_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: bool
                            out9_: _dafny.Seq
                            out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_nextSql_)
                            d_13_appendedGenerated_ = out7_
                            d_14_appendedInside_ = out8_
                            d_15_appendedCurrent_ = out9_
                            generated = d_13_appendedGenerated_
                            insideConstrainedOut = d_14_appendedInside_
                            currentConstrainedOut = d_15_appendedCurrent_
                    pass
            pass
        cost = d_3_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

