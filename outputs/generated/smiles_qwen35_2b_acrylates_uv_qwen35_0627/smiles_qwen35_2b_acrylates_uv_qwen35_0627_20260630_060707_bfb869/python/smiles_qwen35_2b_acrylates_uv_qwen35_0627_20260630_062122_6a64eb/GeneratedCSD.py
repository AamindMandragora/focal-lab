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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one valid SMILES string for a novel acrylate ester molecule. The SMILES must contain the acrylate group C=CC(=O)O as a substructure. A valid acrylate SMILES starts with C=CC(=O)O followed by a carbon chain (example: C=CC(=O)OCC, C=CC(=O)OCCO, C=CC(=O)OC(C)C, C=CC(=O)OCCCC). Output ONLY the SMILES string. No explanation, no Markdown, no surrounding text. Do not copy examples from the prompt verbatim."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_acrylateCoreTokens_: _dafny.Seq
        d_2_acrylateCoreTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "c")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "n")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N"))])
        d_3_steps_: int
        d_3_steps_ = 0
        with _dafny.label("0"):
            while (d_3_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_og_: _dafny.Seq
                        d_5_oi_: bool
                        d_6_oc_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_4_og_ = out0_
                        d_5_oi_ = out1_
                        d_6_oc_ = out2_
                        generated = d_4_og_
                        insideConstrainedOut = d_5_oi_
                        currentConstrainedOut = d_6_oc_
                        d_3_steps_ = (d_3_steps_) + (1)
                    elif True:
                        d_7_cg_: _dafny.Seq
                        d_8_ci_: bool
                        d_9_cc_: _dafny.Seq
                        d_10_closed_: bool
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out6_: bool
                        out3_, out4_, out5_, out6_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_7_cg_ = out3_
                        d_8_ci_ = out4_
                        d_9_cc_ = out5_
                        d_10_closed_ = out6_
                        d_3_steps_ = (d_3_steps_) + (1)
                        if d_10_closed_:
                            generated = d_7_cg_
                            insideConstrainedOut = d_8_ci_
                            currentConstrainedOut = d_9_cc_
                            raise _dafny.Break("0")
                        elif True:
                            if (d_3_steps_) < (maxSteps):
                                d_11_constrainedPrompt_: _dafny.Seq
                                d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                                d_12_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                                d_13_spanLen_: int
                                d_13_spanLen_ = len(currentConstrainedOut)
                                if (d_13_spanLen_) < (12):
                                    out7_: _dafny.Seq
                                    out7_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, d_2_acrylateCoreTokens_, _dafny.BigRational('6e0'), eosToken)
                                    d_12_next_ = out7_
                                elif True:
                                    out8_: _dafny.Seq
                                    out8_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                                    d_12_next_ = out8_
                                d_3_steps_ = (d_3_steps_) + (1)
                                if (d_12_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_14_ag_: _dafny.Seq
                                    d_15_ai_: bool
                                    d_16_ac_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                                    d_14_ag_ = out9_
                                    d_15_ai_ = out10_
                                    d_16_ac_ = out11_
                                    generated = d_14_ag_
                                    insideConstrainedOut = d_15_ai_
                                    currentConstrainedOut = d_16_ac_
                    pass
            pass
        if ((insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_3_steps_) < (maxSteps)):
            d_17_cg_: _dafny.Seq
            d_18_ci_: bool
            d_19_cc_: _dafny.Seq
            out12_: _dafny.Seq
            out13_: bool
            out14_: _dafny.Seq
            out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
            d_17_cg_ = out12_
            d_18_ci_ = out13_
            d_19_cc_ = out14_
            generated = d_17_cg_
            insideConstrainedOut = d_18_ci_
            currentConstrainedOut = d_19_cc_
            d_3_steps_ = (d_3_steps_) + (1)
        cost = d_3_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

