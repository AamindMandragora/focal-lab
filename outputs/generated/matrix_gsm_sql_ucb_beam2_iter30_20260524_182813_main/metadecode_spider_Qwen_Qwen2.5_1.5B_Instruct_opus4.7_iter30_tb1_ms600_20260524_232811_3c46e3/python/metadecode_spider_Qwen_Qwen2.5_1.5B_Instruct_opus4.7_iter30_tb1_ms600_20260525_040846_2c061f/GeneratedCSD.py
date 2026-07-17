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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly: SQL: <<QUERY>>. Emit 'SQL:' then a space then '<<' then a single SQLite SELECT statement using only the schema names from the prompt, then '>>'. Keep the query short - do not add WHERE, JOIN, GROUP BY, ORDER BY, or LIMIT clauses unless the question explicitly requires them. No markdown. No code fences. No commentary before or after.")))
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
                                d_10_hasOpen_: bool
                                d_10_hasOpen_ = False
                                if (d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_10_hasOpen_ = True
                                elif (len(d_9_next_)) >= (2):
                                    d_11_k_: int
                                    d_11_k_ = 0
                                    with _dafny.label("0_0_1_1_1_0_0"):
                                        while ((d_11_k_) + (2)) <= (len(d_9_next_)):
                                            with _dafny.c_label("0_0_1_1_1_0_0"):
                                                if (((d_9_next_)[d_11_k_]) == (_dafny.CodePoint('<'))) and (((d_9_next_)[(d_11_k_) + (1)]) == (_dafny.CodePoint('<'))):
                                                    d_10_hasOpen_ = True
                                                    raise _dafny.Break("0_0_1_1_1_0_0")
                                                d_11_k_ = (d_11_k_) + (1)
                                                pass
                                        pass
                                if d_10_hasOpen_:
                                    d_12_enteredG_: _dafny.Seq
                                    d_13_enteredI_: bool
                                    d_14_enteredC_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_12_enteredG_ = out4_
                                    d_13_enteredI_ = out5_
                                    d_14_enteredC_ = out6_
                                    generated = d_12_enteredG_
                                    insideConstrainedOut = d_13_enteredI_
                                    currentConstrainedOut = d_14_enteredC_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_15_closedG_: _dafny.Seq
                        d_16_closedI_: bool
                        d_17_closedC_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_15_closedG_ = out7_
                        d_16_closedI_ = out8_
                        d_17_closedC_ = out9_
                        generated = d_15_closedG_
                        insideConstrainedOut = d_16_closedI_
                        currentConstrainedOut = d_17_closedC_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif ((len(currentConstrainedOut)) >= (d_4_spanLengthCap_)) or (((d_1_steps_) + (1)) >= (maxSteps)):
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                        insideConstrainedOut = False
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_next_: _dafny.Seq
                        out10_: _dafny.Seq
                        out10_ = (d_0_helpers_).AdaptiveConstrainedStepWithPenalties(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), d_5_penaltyTokens_, _dafny.BigRational('8e0'), 12, eosToken)
                        d_19_next_ = out10_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_19_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_20_appendedG_: _dafny.Seq
                            d_21_appendedI_: bool
                            d_22_appendedC_: _dafny.Seq
                            out11_: _dafny.Seq
                            out12_: bool
                            out13_: _dafny.Seq
                            out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                            d_20_appendedG_ = out11_
                            d_21_appendedI_ = out12_
                            d_22_appendedC_ = out13_
                            generated = d_20_appendedG_
                            insideConstrainedOut = d_21_appendedI_
                            currentConstrainedOut = d_22_appendedC_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

