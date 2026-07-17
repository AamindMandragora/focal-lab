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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single valid SMILES string for a novel acrylate ester molecule. Acrylates contain the CH2=CH-C(=O)-O- group (C=CC(=O)O...). Output ONLY the SMILES string. Do NOT copy these exemplars: C=CC(=O)OCC, C=CC(=O)OCCCO, C=CC(=O)OC(C)C, C=CC(=O)OCCO, C=CC(=O)OCC(C)C, C=CC(=O)OCCCCC, C=CC(=O)OC1CCCCC1, C=CC(=O)OCC(O)CO. Generate a NEW acrylate such as C=CC(=O)OCCC or C=CC(=O)OCCCCCC or C=CC(=O)OCC(CC)CC."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        if (d_2_steps_) < (maxSteps):
            d_3_og_: _dafny.Seq
            d_4_oi_: bool
            d_5_oc_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_3_og_ = out0_
            d_4_oi_ = out1_
            d_5_oc_ = out2_
            generated = d_3_og_
            insideConstrainedOut = d_4_oi_
            currentConstrainedOut = d_5_oc_
            d_2_steps_ = (d_2_steps_) + (1)
        with _dafny.label("0"):
            while (((d_2_steps_) < (maxSteps)) and (insideConstrainedOut)) and (not((parser).IsCompletePrefix(currentConstrainedOut))):
                with _dafny.c_label("0"):
                    d_6_constrainedPrompt_: _dafny.Seq
                    d_6_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_7_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_6_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                    d_7_next_ = out3_
                    d_2_steps_ = (d_2_steps_) + (1)
                    if (d_7_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        d_8_ag_: _dafny.Seq
                        d_9_ai_: bool
                        d_10_ac_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_7_next_)
                        d_8_ag_ = out4_
                        d_9_ai_ = out5_
                        d_10_ac_ = out6_
                        generated = d_8_ag_
                        insideConstrainedOut = d_9_ai_
                        currentConstrainedOut = d_10_ac_
                    pass
            pass
        if (((d_2_steps_) < (maxSteps)) and (insideConstrainedOut)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
            d_11_cg_: _dafny.Seq
            d_12_ci_: bool
            d_13_cc_: _dafny.Seq
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
            d_11_cg_ = out7_
            d_12_ci_ = out8_
            d_13_cc_ = out9_
            generated = d_11_cg_
            insideConstrainedOut = d_12_ci_
            currentConstrainedOut = d_13_cc_
            d_2_steps_ = (d_2_steps_) + (1)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

