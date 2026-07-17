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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a valid SMILES string for an acrylate molecule. Acrylates contain the acryloyl group C=CC(=O)O. Output only the SMILES string with no explanation.")))
        if (maxSteps) == (0):
            pass
        elif not(insideConstrained):
            d_1_constrainedGenerated_: _dafny.Seq
            d_2_terminatedByEos_: bool
            out0_: _dafny.Seq
            out1_: bool
            out0_, out1_ = (d_0_helpers_).ConstrainedGeneration(lm, parser, prompt, maxSteps, eosToken)
            d_1_constrainedGenerated_ = out0_
            d_2_terminatedByEos_ = out1_
            generated = (generatedPrefix) + (d_1_constrainedGenerated_)
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            cost = maxSteps
        elif True:
            d_3_steps_: int
            d_3_steps_ = 0
            with _dafny.label("1_1_0"):
                while (d_3_steps_) < (maxSteps):
                    with _dafny.c_label("1_1_0"):
                        if not(insideConstrainedOut):
                            raise _dafny.Break("1_1_0")
                        elif (parser).IsCompletePrefix(currentConstrainedOut):
                            d_4_cg_: _dafny.Seq
                            d_5_ci_: bool
                            d_6_cc_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_4_cg_ = out2_
                            d_5_ci_ = out3_
                            d_6_cc_ = out4_
                            generated = d_4_cg_
                            insideConstrainedOut = d_5_ci_
                            currentConstrainedOut = d_6_cc_
                            d_3_steps_ = (d_3_steps_) + (1)
                            raise _dafny.Break("1_1_0")
                        elif True:
                            d_7_cp_: _dafny.Seq
                            d_7_cp_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_8_next_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_7_cp_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_8_next_ = out5_
                            d_3_steps_ = (d_3_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("1_1_0")
                            elif True:
                                d_9_ag_: _dafny.Seq
                                d_10_ai_: bool
                                d_11_ac_: _dafny.Seq
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: _dafny.Seq
                                out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_8_next_)
                                d_9_ag_ = out6_
                                d_10_ai_ = out7_
                                d_11_ac_ = out8_
                                generated = d_9_ag_
                                insideConstrainedOut = d_10_ai_
                                currentConstrainedOut = d_11_ac_
                        pass
                pass
            cost = d_3_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

