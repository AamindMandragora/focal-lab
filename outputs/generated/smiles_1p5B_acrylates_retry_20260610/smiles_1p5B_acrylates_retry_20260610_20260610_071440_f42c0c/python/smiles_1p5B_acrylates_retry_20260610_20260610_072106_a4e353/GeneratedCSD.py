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
        d_1_steps_: int
        d_1_steps_ = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "You are generating SMILES strings for acrylate monomers. Acrylates MUST contain the vinyl ester group. A valid acrylate SMILES always has multiple atoms. Do NOT output a single atom. Generate multi-atom SMILES like: C=CC(=O)OCC, C=CC(=O)OCCO, C=C(C)C(=O)OCC, C=CC(=O)OC(C)C, C=CC(=O)OCCOCCO. Start with C= to begin the vinyl group.")))
        if ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
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
        d_5_minLength_: int
        d_5_minLength_ = 8
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    d_6_constrainedPrompt_: _dafny.Seq
                    d_6_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_7_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_6_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 20, eosToken)
                    d_7_next_ = out3_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_7_next_) == (eosToken):
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_8_fg_: _dafny.Seq
                            d_9_fi_: bool
                            d_10_fc_: _dafny.Seq
                            out4_: _dafny.Seq
                            out5_: bool
                            out6_: _dafny.Seq
                            out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_8_fg_ = out4_
                            d_9_fi_ = out5_
                            d_10_fc_ = out6_
                            generated = d_8_fg_
                            insideConstrainedOut = d_9_fi_
                            currentConstrainedOut = d_10_fc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif (d_1_steps_) < (maxSteps):
                            d_11_rg_: _dafny.Seq
                            d_12_rc_: _dafny.Seq
                            out7_: _dafny.Seq
                            out8_: _dafny.Seq
                            out7_, out8_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_11_rg_ = out7_
                            d_12_rc_ = out8_
                            generated = d_11_rg_
                            currentConstrainedOut = d_12_rc_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_13_fg_: _dafny.Seq
                                d_14_fi_: bool
                                d_15_fc_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_13_fg_ = out9_
                                d_14_fi_ = out10_
                                d_15_fc_ = out11_
                                generated = d_13_fg_
                                insideConstrainedOut = d_14_fi_
                                currentConstrainedOut = d_15_fc_
                                d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                        currentConstrainedOut = (currentConstrainedOut) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (d_5_minLength_)):
                            if (d_1_steps_) < (maxSteps):
                                d_16_fg_: _dafny.Seq
                                d_17_fi_: bool
                                d_18_fc_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: bool
                                out14_: _dafny.Seq
                                out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_16_fg_ = out12_
                                d_17_fi_ = out13_
                                d_18_fc_ = out14_
                                generated = d_16_fg_
                                insideConstrainedOut = d_17_fi_
                                currentConstrainedOut = d_18_fc_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_19_fg_: _dafny.Seq
                d_20_fi_: bool
                d_21_fc_: _dafny.Seq
                out15_: _dafny.Seq
                out16_: bool
                out17_: _dafny.Seq
                out15_, out16_, out17_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_19_fg_ = out15_
                d_20_fi_ = out16_
                d_21_fc_ = out17_
                generated = d_19_fg_
                insideConstrainedOut = d_20_fi_
                currentConstrainedOut = d_21_fc_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_22_rg_: _dafny.Seq
                d_23_rc_: _dafny.Seq
                out18_: _dafny.Seq
                out19_: _dafny.Seq
                out18_, out19_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                d_22_rg_ = out18_
                d_23_rc_ = out19_
                generated = d_22_rg_
                currentConstrainedOut = d_23_rc_
                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                    d_24_fg_: _dafny.Seq
                    d_25_fi_: bool
                    d_26_fc_: _dafny.Seq
                    out20_: _dafny.Seq
                    out21_: bool
                    out22_: _dafny.Seq
                    out20_, out21_, out22_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                    d_24_fg_ = out20_
                    d_25_fi_ = out21_
                    d_26_fc_ = out22_
                    generated = d_24_fg_
                    insideConstrainedOut = d_25_fi_
                    currentConstrainedOut = d_26_fc_
                    d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

