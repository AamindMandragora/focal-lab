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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a valid SMILES string for a novel acrylate ester molecule. Acrylates contain the CH2=CH-C(=O)-O- core. Examples: C=CC(=O)OCC, C=CC(=O)OCCO, C=CC(=O)OC(C)C. Output only the SMILES string."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        if (maxSteps) == (0):
            pass
        elif insideConstrainedOut:
            d_2_steps_: int
            d_2_steps_ = 0
            with _dafny.label("1_0_0"):
                while (d_2_steps_) < (maxSteps):
                    with _dafny.c_label("1_0_0"):
                        d_3_isComplete_: bool
                        d_3_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_3_isComplete_:
                            d_4_cg_: _dafny.Seq
                            d_5_ci_: bool
                            d_6_cc_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_4_cg_ = out0_
                            d_5_ci_ = out1_
                            d_6_cc_ = out2_
                            generated = d_4_cg_
                            insideConstrainedOut = d_5_ci_
                            currentConstrainedOut = d_6_cc_
                            d_2_steps_ = (d_2_steps_) + (1)
                            raise _dafny.Break("1_0_0")
                        elif True:
                            d_7_constrainedPrompt_: _dafny.Seq
                            d_7_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_8_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 20, eosToken)
                            d_8_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("1_0_0")
                            elif True:
                                d_9_ag_: _dafny.Seq
                                d_10_ai_: bool
                                d_11_ac_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_8_next_)
                                d_9_ag_ = out4_
                                d_10_ai_ = out5_
                                d_11_ac_ = out6_
                                generated = d_9_ag_
                                insideConstrainedOut = d_10_ai_
                                currentConstrainedOut = d_11_ac_
                        pass
                pass
            cost = d_2_steps_
        elif True:
            d_12_constrainedOutput_: _dafny.Seq
            d_13_terminatedByEos_: bool
            out7_: _dafny.Seq
            out8_: bool
            out7_, out8_ = (d_0_helpers_).ConstrainedGeneration(lm, parser, (prompt) + (generatedPrefix), maxSteps, eosToken)
            d_12_constrainedOutput_ = out7_
            d_13_terminatedByEos_ = out8_
            generated = (generatedPrefix) + (d_12_constrainedOutput_)
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            d_14_outputLen_: int
            d_14_outputLen_ = len(d_12_constrainedOutput_)
            if d_13_terminatedByEos_:
                cost = (d_14_outputLen_) + (1)
                if (cost) > (maxSteps):
                    cost = maxSteps
            elif True:
                cost = d_14_outputLen_
                if ((cost) == (0)) and ((maxSteps) > (0)):
                    cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

