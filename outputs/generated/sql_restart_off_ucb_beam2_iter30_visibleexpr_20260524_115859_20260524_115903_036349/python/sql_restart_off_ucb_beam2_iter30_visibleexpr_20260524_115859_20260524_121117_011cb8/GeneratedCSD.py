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
        if (maxSteps) == (0):
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Write exactly one SQL query that answers the question using the schema. Strict style: do NOT use table aliases like T1, T2, or AS. Reference every column as table.column using the real table name. When filtering by a value mentioned in the question, filter directly on the column that holds that value (do not add unnecessary joins). Use lowercase SQL keywords. Place the full query between << and >> and emit nothing else.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_og_: _dafny.Seq
                        d_3_oi_: bool
                        d_4_oc_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_2_og_ = out0_
                        d_3_oi_ = out1_
                        d_4_oc_ = out2_
                        generated = d_2_og_
                        insideConstrainedOut = d_3_oi_
                        currentConstrainedOut = d_4_oc_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_5_cg_: _dafny.Seq
                        d_6_ci_: bool
                        d_7_cc_: _dafny.Seq
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_5_cg_ = out3_
                        d_6_ci_ = out4_
                        d_7_cc_ = out5_
                        generated = d_5_cg_
                        insideConstrainedOut = d_6_ci_
                        currentConstrainedOut = d_7_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_8_constrainedPrompt_: _dafny.Seq
                        d_8_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_9_next_: _dafny.Seq
                        d_10_wasConstrained_: bool
                        out6_: _dafny.Seq
                        out7_: bool
                        out6_, out7_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_9_next_ = out6_
                        d_10_wasConstrained_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_9_next_) == (eosToken):
                            raise _dafny.Break("0")
                        d_11_validNext_: bool
                        out8_: bool
                        out8_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_9_next_)
                        d_11_validNext_ = out8_
                        if d_11_validNext_:
                            d_12_ag_: _dafny.Seq
                            d_13_ai_: bool
                            d_14_ac_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                            d_12_ag_ = out9_
                            d_13_ai_ = out10_
                            d_14_ac_ = out11_
                            generated = d_12_ag_
                            insideConstrainedOut = d_13_ai_
                            currentConstrainedOut = d_14_ac_
                        elif True:
                            raise _dafny.Break("0")
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

