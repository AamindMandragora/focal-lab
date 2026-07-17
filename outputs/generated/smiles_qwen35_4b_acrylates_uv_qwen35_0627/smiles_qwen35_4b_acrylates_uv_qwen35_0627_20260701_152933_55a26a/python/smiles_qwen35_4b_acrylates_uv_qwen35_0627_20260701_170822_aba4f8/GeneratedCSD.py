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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CRITICAL: Generate exactly one valid SMILES for an acrylate ester. The SMILES MUST contain the acrylate vinyl ester core: C=CC(=O)O followed by an alkyl group R. Examples: C=CC(=O)OCC (ethyl acrylate), C=CC(=O)OCCC (propyl acrylate), C=CC(=O)OC(C)C (isopropyl acrylate), C=CC(=O)OCCCC (butyl acrylate). Output ONLY the SMILES string."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        if (d_2_steps_) < (maxSteps):
            d_3_gO_: _dafny.Seq
            d_4_iO_: bool
            d_5_cO_: _dafny.Seq
            out0_: _dafny.Seq
            out1_: bool
            out2_: _dafny.Seq
            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_3_gO_ = out0_
            d_4_iO_ = out1_
            d_5_cO_ = out2_
            generated = d_3_gO_
            insideConstrainedOut = d_4_iO_
            currentConstrainedOut = d_5_cO_
            d_2_steps_ = (d_2_steps_) + (1)
        d_6_acrylateCoreTokens_: _dafny.Seq
        d_6_acrylateCoreTokens_ = _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "(")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ")")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O"))])
        d_7_coreIdx_: int
        d_7_coreIdx_ = 0
        with _dafny.label("0"):
            while (((d_7_coreIdx_) < (len(d_6_acrylateCoreTokens_))) and ((d_2_steps_) < (maxSteps))) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    d_8_targetToken_: _dafny.Seq
                    d_8_targetToken_ = (d_6_acrylateCoreTokens_)[d_7_coreIdx_]
                    d_9_isValid_: bool
                    out3_: bool
                    out3_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_8_targetToken_)
                    d_9_isValid_ = out3_
                    if d_9_isValid_:
                        d_10_constrainedPrompt_: _dafny.Seq
                        d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_11_tokensToBoost_: _dafny.Seq
                        d_11_tokensToBoost_ = _dafny.SeqWithoutIsStrInference([d_8_targetToken_])
                        d_12_next_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, d_11_tokensToBoost_, _dafny.BigRational('2e1'), eosToken)
                        d_12_next_ = out4_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_12_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_13_isComplete_: bool
                            d_13_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_13_isComplete_:
                                raise _dafny.Break("0")
                            elif True:
                                d_14_gA_: _dafny.Seq
                                d_15_iA_: bool
                                d_16_cA_: _dafny.Seq
                                out5_: _dafny.Seq
                                out6_: bool
                                out7_: _dafny.Seq
                                out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                                d_14_gA_ = out5_
                                d_15_iA_ = out6_
                                d_16_cA_ = out7_
                                generated = d_14_gA_
                                insideConstrainedOut = d_15_iA_
                                currentConstrainedOut = d_16_cA_
                                d_7_coreIdx_ = (d_7_coreIdx_) + (1)
                    elif True:
                        d_7_coreIdx_ = (d_7_coreIdx_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            d_17_closeBudget_: int
            d_17_closeBudget_ = (maxSteps) - (d_2_steps_)
            d_18_cg_: _dafny.Seq
            d_19_ci_: bool
            d_20_cc_: _dafny.Seq
            out8_: _dafny.Seq
            out9_: bool
            out10_: _dafny.Seq
            out8_, out9_, out10_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_closeBudget_)
            d_18_cg_ = out8_
            d_19_ci_ = out9_
            d_20_cc_ = out10_
            generated = d_18_cg_
            insideConstrainedOut = d_19_ci_
            currentConstrainedOut = d_20_cc_
            d_2_steps_ = maxSteps
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

