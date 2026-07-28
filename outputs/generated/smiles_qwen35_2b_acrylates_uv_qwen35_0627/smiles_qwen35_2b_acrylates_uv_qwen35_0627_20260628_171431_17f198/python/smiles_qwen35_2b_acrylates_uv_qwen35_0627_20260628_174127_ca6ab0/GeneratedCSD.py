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
        out0_: _dafny.Seq
        out1_: bool
        out2_: _dafny.Seq
        out3_: int
        out0_, out1_, out2_, out3_ = default__.AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken)
        generated = out0_
        insideConstrainedOut = out1_
        currentConstrainedOut = out2_
        cost = out3_
        if ((maxSteps) > (0)) and ((cost) <= (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

    @staticmethod
    def AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate one valid SMILES string for a novel acrylate ester. Acrylates contain the CH2=CH-C(=O)-O- group. Output only the SMILES string.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if (not(insideConstrainedOut)) and ((maxSteps) > (0)):
            d_2_constrainedResult_: _dafny.Seq
            d_3_terminatedByEos_: bool
            out0_: _dafny.Seq
            out1_: bool
            out0_, out1_ = (d_0_helpers_).ConstrainedGeneration(lm, parser, prompt, maxSteps, eosToken)
            d_2_constrainedResult_ = out0_
            d_3_terminatedByEos_ = out1_
            generated = (generatedPrefix) + (d_2_constrainedResult_)
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            insideConstrainedOut = False
            d_1_steps_ = maxSteps
        elif (insideConstrained) and ((maxSteps) > (0)):
            with _dafny.label("1_0_0"):
                while (((d_1_steps_) < (maxSteps)) and (insideConstrainedOut)) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                    with _dafny.c_label("1_0_0"):
                        d_4_stableLen_: int
                        d_4_stableLen_ = (len(generated)) - (len(currentConstrainedOut))
                        d_5_constrainedPrompt_: _dafny.Seq
                        d_5_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:d_4_stableLen_:]))
                        d_6_next_: _dafny.Seq
                        out2_: _dafny.Seq
                        out2_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_5_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_6_next_ = out2_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_6_next_) == (eosToken):
                            raise _dafny.Break("1_0_0")
                        elif True:
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_6_next_)
                            generated = out3_
                            insideConstrainedOut = out4_
                            currentConstrainedOut = out5_
                        pass
                pass
            if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
                d_7_cg_: _dafny.Seq
                d_8_ci_: bool
                d_9_cc_: _dafny.Seq
                d_10_closed_: bool
                out6_: _dafny.Seq
                out7_: bool
                out8_: _dafny.Seq
                out9_: bool
                out6_, out7_, out8_, out9_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                d_7_cg_ = out6_
                d_8_ci_ = out7_
                d_9_cc_ = out8_
                d_10_closed_ = out9_
                d_1_steps_ = (d_1_steps_) + (1)
                if d_10_closed_:
                    generated = d_7_cg_
                    insideConstrainedOut = d_8_ci_
                    currentConstrainedOut = d_9_cc_
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

