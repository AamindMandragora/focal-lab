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
        d_1_steps_: int
        d_1_steps_ = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "TASK: Output exactly one valid SMILES string for a NEW isocyanate molecule. Isocyanates have the functional group -N=C=O. The SMILES must contain the substructure N=C=O or O=C=N. Examples of valid isocyanate SMILES that you must NOT copy: CN=C=O, O=C=NC, O=C=NCC, CCN=C=O. Generate a different isocyanate SMILES, such as O=C=NCCC, O=C=NC(C)C, O=C=Nc1ccccc1, O=C=NCC(C)C. Output ONLY the SMILES. No explanation.")))
        d_2_preambleBudget_: int
        d_2_preambleBudget_ = 15
        if ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
            d_3_actualPreamble_: int
            d_3_actualPreamble_ = d_2_preambleBudget_
            if ((maxSteps) - (d_1_steps_)) < (d_2_preambleBudget_):
                d_3_actualPreamble_ = (maxSteps) - (d_1_steps_)
            if (d_3_actualPreamble_) > (0):
                d_4_chunkedGenerated_: _dafny.Seq
                d_5_stoppedOnOpen_: bool
                d_6_stoppedOnEos_: bool
                d_7_chunkSteps_: int
                out0_: _dafny.Seq
                out1_: bool
                out2_: bool
                out3_: int
                out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_3_actualPreamble_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                d_4_chunkedGenerated_ = out0_
                d_5_stoppedOnOpen_ = out1_
                d_6_stoppedOnEos_ = out2_
                d_7_chunkSteps_ = out3_
                generated = d_4_chunkedGenerated_
                d_1_steps_ = (d_1_steps_) + (d_7_chunkSteps_)
                if d_6_stoppedOnEos_:
                    cost = d_1_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                if d_5_stoppedOnOpen_:
                    d_8_og_: _dafny.Seq
                    d_9_oi_: bool
                    d_10_oc_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: _dafny.Seq
                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                    d_8_og_ = out4_
                    d_9_oi_ = out5_
                    d_10_oc_ = out6_
                    generated = d_8_og_
                    insideConstrainedOut = d_9_oi_
                    currentConstrainedOut = d_10_oc_
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_11_og_: _dafny.Seq
            d_12_oi_: bool
            d_13_oc_: _dafny.Seq
            out7_: _dafny.Seq
            out8_: bool
            out9_: _dafny.Seq
            out7_, out8_, out9_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_11_og_ = out7_
            d_12_oi_ = out8_
            d_13_oc_ = out9_
            generated = d_11_og_
            insideConstrainedOut = d_12_oi_
            currentConstrainedOut = d_13_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        d_14_isocyanateGroups_: _dafny.Seq
        d_14_isocyanateGroups_ = (validTokenGroups) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "N")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "O")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "C"))])]))
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    if (parser).IsCompletePrefix(currentConstrainedOut):
                        d_15_cg_: _dafny.Seq
                        d_16_ci_: bool
                        d_17_cc_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_15_cg_ = out10_
                        d_16_ci_ = out11_
                        d_17_cc_ = out12_
                        generated = d_15_cg_
                        insideConstrainedOut = d_16_ci_
                        currentConstrainedOut = d_17_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_tokenCount_: int
                        d_19_tokenCount_ = len(currentConstrainedOut)
                        d_20_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if (d_19_tokenCount_) < (8):
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, d_14_isocyanateGroups_, _dafny.BigRational('5e0'), 20, eosToken)
                            d_20_next_ = out13_
                        elif True:
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('15e-1'), eosToken)
                            d_20_next_ = out14_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_20_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_21_ag_: _dafny.Seq
                            d_22_ai_: bool
                            d_23_ac_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_20_next_)
                            d_21_ag_ = out15_
                            d_22_ai_ = out16_
                            d_23_ac_ = out17_
                            generated = d_21_ag_
                            insideConstrainedOut = d_22_ai_
                            currentConstrainedOut = d_23_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_24_closeBudget_: int
            d_24_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_25_cg_: _dafny.Seq
            d_26_ci_: bool
            d_27_cc_: _dafny.Seq
            out18_: _dafny.Seq
            out19_: bool
            out20_: _dafny.Seq
            out18_, out19_, out20_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_24_closeBudget_)
            d_25_cg_ = out18_
            d_26_ci_ = out19_
            d_27_cc_ = out20_
            generated = d_25_cg_
            insideConstrainedOut = d_26_ci_
            currentConstrainedOut = d_27_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

