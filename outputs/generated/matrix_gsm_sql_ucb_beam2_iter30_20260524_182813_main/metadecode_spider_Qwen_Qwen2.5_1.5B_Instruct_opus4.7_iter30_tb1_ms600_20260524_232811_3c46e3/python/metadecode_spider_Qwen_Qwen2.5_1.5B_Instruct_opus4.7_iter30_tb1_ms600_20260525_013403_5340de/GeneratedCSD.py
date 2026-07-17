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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output format: exactly `SQL: <<QUERY>>` on one line. Emit `SQL:`, one space, `<<`, then one SQLite SELECT statement, then `>>`. No semicolon, no markdown, no code fences, no commentary. CRITICAL schema rules: use ONLY the exact table names and column names that appear verbatim in the schema given in the prompt; never rename, abbreviate, pluralize, or invent. When a needed column lives in another table, JOIN through the foreign keys shown in the schema. Prefer the shortest correct query and avoid unnecessary JOINs. Example: for question 'How many singers are there?' with schema `singer(singer_id, name, country)`, the answer is `SQL: <<SELECT COUNT(*) FROM singer>>`. Example: for question 'List the names of singers from France in descending order of age' with schema `singer(singer_id, name, country, age)`, the answer is `SQL: <<SELECT name FROM singer WHERE country = 'France' ORDER BY age DESC>>`.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_unconstrainedCount_: int
        d_2_unconstrainedCount_ = 0
        d_3_preambleCap_: int
        d_3_preambleCap_ = 3
        d_4_spanLengthCap_: int
        d_4_spanLengthCap_ = 80
        d_5_penaltyTokens_: _dafny.Seq
        d_5_penaltyTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_2_unconstrainedCount_) >= (d_3_preambleCap_):
                            d_6_openedG_: _dafny.Seq
                            d_7_openedI_: bool
                            d_8_openedC_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_openedG_ = out0_
                            d_7_openedI_ = out1_
                            d_8_openedC_ = out2_
                            generated = d_6_openedG_
                            insideConstrainedOut = d_7_openedI_
                            currentConstrainedOut = d_8_openedC_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_9_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_unconstrainedCount_ = (d_2_unconstrainedCount_) + (1)
                            if (d_9_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                d_10_naturalOpen_: bool
                                d_10_naturalOpen_ = ((d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))) or ((((len(d_9_next_)) >= (2)) and ((len(d_9_next_)) <= (4))) and ((_dafny.SeqWithoutIsStrInference((d_9_next_)[(len(d_9_next_)) - (2)::])) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))))
                                if d_10_naturalOpen_:
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_11_closedG_: _dafny.Seq
                        d_12_closedI_: bool
                        d_13_closedC_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_11_closedG_ = out4_
                        d_12_closedI_ = out5_
                        d_13_closedC_ = out6_
                        generated = d_11_closedG_
                        insideConstrainedOut = d_12_closedI_
                        currentConstrainedOut = d_13_closedC_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif ((len(currentConstrainedOut)) >= (d_4_spanLengthCap_)) or (((d_1_steps_) + (1)) >= (maxSteps)):
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                        insideConstrainedOut = False
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_14_constrainedPrompt_: _dafny.Seq
                        d_14_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_15_next_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_14_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_5_penaltyTokens_, _dafny.BigRational('8e0'), 12, eosToken)
                        d_15_next_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_15_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_16_appendedG_: _dafny.Seq
                            d_17_appendedI_: bool
                            d_18_appendedC_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                            d_16_appendedG_ = out8_
                            d_17_appendedI_ = out9_
                            d_18_appendedC_ = out10_
                            generated = d_16_appendedG_
                            insideConstrainedOut = d_17_appendedI_
                            currentConstrainedOut = d_18_appendedC_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

