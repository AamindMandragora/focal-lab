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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output a single valid SMILES string for a novel acrylate ester molecule. Acrylates have the core structure C=CC(=O)O followed by an ester group. Output ONLY the SMILES string with no explanation."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
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
                        elif True:
                            d_7_constrainedPrompt_: _dafny.Seq
                            d_7_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_8_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_8_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_9_isValid_: bool
                                out4_: bool
                                out4_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_8_next_)
                                d_9_isValid_ = out4_
                                if d_9_isValid_:
                                    d_10_notComplete_: bool
                                    d_10_notComplete_ = not((parser).IsCompletePrefix(currentConstrainedOut))
                                    if d_10_notComplete_:
                                        d_11_ag_: _dafny.Seq
                                        d_12_ai_: bool
                                        d_13_ac_: _dafny.Seq
                                        out5_: _dafny.Seq
                                        out6_: bool
                                        out7_: _dafny.Seq
                                        out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_8_next_)
                                        d_11_ag_ = out5_
                                        d_12_ai_ = out6_
                                        d_13_ac_ = out7_
                                        generated = d_11_ag_
                                        insideConstrainedOut = d_12_ai_
                                        currentConstrainedOut = d_13_ac_
                    elif True:
                        d_14_constrainedPrompt_: _dafny.Seq
                        d_14_constrainedPrompt_ = (prompt) + (generated)
                        d_15_next_: _dafny.Seq
                        out8_: _dafny.Seq
                        out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_14_constrainedPrompt_, _dafny.SeqWithoutIsStrInference([]), eosToken)
                        d_15_next_ = out8_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_15_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_16_isValid_: bool
                            out9_: bool
                            out9_ = (d_0_helpers_).IsTokenValidNext(parser, _dafny.SeqWithoutIsStrInference([]), d_15_next_)
                            d_16_isValid_ = out9_
                            if d_16_isValid_:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_15_next_]))
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

