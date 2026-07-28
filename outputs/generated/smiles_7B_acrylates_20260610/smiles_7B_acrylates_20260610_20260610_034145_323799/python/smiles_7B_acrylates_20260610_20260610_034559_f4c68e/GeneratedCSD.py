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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single valid SMILES string for an acrylate molecule. Acrylates contain the C=CC(=O)O motif. Output only the SMILES string with no explanation.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if not(insideConstrainedOut):
            insideConstrainedOut = True
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_2_cg_: _dafny.Seq
                        d_3_ci_: bool
                        d_4_cc_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_2_cg_ = out0_
                        d_3_ci_ = out1_
                        d_4_cc_ = out2_
                        generated = d_2_cg_
                        insideConstrainedOut = d_3_ci_
                        currentConstrainedOut = d_4_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_5_cp_: _dafny.Seq
                        d_5_cp_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_6_next_: _dafny.Seq
                        out3_: _dafny.Seq
                        out3_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_5_cp_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 20, eosToken)
                        d_6_next_ = out3_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_6_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            if not((parser).IsCompletePrefix(currentConstrainedOut)):
                                d_7_ag_: _dafny.Seq
                                d_8_ai_: bool
                                d_9_ac_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_6_next_)
                                d_7_ag_ = out4_
                                d_8_ai_ = out5_
                                d_9_ac_ = out6_
                                generated = d_7_ag_
                                insideConstrainedOut = d_8_ai_
                                currentConstrainedOut = d_9_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

