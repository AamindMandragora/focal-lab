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
        d_2_flatTokens_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_2_flatTokens_ = out0_
        d_3_seedIndex_: int
        d_3_seedIndex_ = _dafny.euclidian_modulus(len(d_2_flatTokens_), 50)
        d_4_guidance_: _dafny.Seq
        d_4_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate ONLY this SMILES string for a chain_extender molecule (no explanation, no other text): "))
        if (d_3_seedIndex_) == (0):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NCCN")))
        elif (d_3_seedIndex_) == (1):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NCCCN")))
        elif (d_3_seedIndex_) == (2):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NCCCCN")))
        elif (d_3_seedIndex_) == (3):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NCCCCCN")))
        elif (d_3_seedIndex_) == (4):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NCCOCCN")))
        elif (d_3_seedIndex_) == (5):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NCCOCN")))
        elif (d_3_seedIndex_) == (6):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OCCO")))
        elif (d_3_seedIndex_) == (7):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OCCCO")))
        elif (d_3_seedIndex_) == (8):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OCCCCO")))
        elif (d_3_seedIndex_) == (9):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OCCCCCO")))
        elif (d_3_seedIndex_) == (10):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OCCOCCO")))
        elif (d_3_seedIndex_) == (11):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OCCCCCCO")))
        elif (d_3_seedIndex_) == (12):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OCCCCCCCO")))
        elif (d_3_seedIndex_) == (13):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OCC(CO)CO")))
        elif (d_3_seedIndex_) == (14):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NCCO")))
        elif (d_3_seedIndex_) == (15):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NCCCO")))
        elif (d_3_seedIndex_) == (16):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NCCCCO")))
        elif (d_3_seedIndex_) == (17):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NCCCCCO")))
        elif (d_3_seedIndex_) == (18):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NC(C)CN")))
        elif (d_3_seedIndex_) == (19):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NC(C)CO")))
        elif (d_3_seedIndex_) == (20):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OC(C)CCO")))
        elif (d_3_seedIndex_) == (21):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OCC(C)CO")))
        elif (d_3_seedIndex_) == (22):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OCC(O)C")))
        elif (d_3_seedIndex_) == (23):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OCCC(C)O")))
        elif (d_3_seedIndex_) == (24):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OC(C)CCCO")))
        elif (d_3_seedIndex_) == (25):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SCCS")))
        elif (d_3_seedIndex_) == (26):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SCCCS")))
        elif (d_3_seedIndex_) == (27):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SCCCCS")))
        elif (d_3_seedIndex_) == (28):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NCc1ccccc1N")))
        elif (d_3_seedIndex_) == (29):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NCc1ccc(N)cc1")))
        elif (d_3_seedIndex_) == (30):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NCCc1ccc(N)cc1")))
        elif (d_3_seedIndex_) == (31):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OCC(O)CO")))
        elif (d_3_seedIndex_) == (32):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NCC(N)C")))
        elif (d_3_seedIndex_) == (33):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OCCCOCCO")))
        elif (d_3_seedIndex_) == (34):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CC(N)CCN")))
        elif (d_3_seedIndex_) == (35):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CC(O)CCO")))
        elif (d_3_seedIndex_) == (36):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CC(N)CO")))
        elif (d_3_seedIndex_) == (37):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CC(O)CN")))
        elif (d_3_seedIndex_) == (38):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NCCNCCN")))
        elif (d_3_seedIndex_) == (39):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OCCOCCCO")))
        elif (d_3_seedIndex_) == (40):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NC1CCCC1N")))
        elif (d_3_seedIndex_) == (41):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OC1CCCCO1")))
        elif (d_3_seedIndex_) == (42):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NC(CC)CO")))
        elif (d_3_seedIndex_) == (43):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NC(CC)CN")))
        elif (d_3_seedIndex_) == (44):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OCC(CC)CO")))
        elif (d_3_seedIndex_) == (45):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NCCOCCN")))
        elif (d_3_seedIndex_) == (46):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OCCOCCCCO")))
        elif (d_3_seedIndex_) == (47):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CC(N)CCCN")))
        elif (d_3_seedIndex_) == (48):
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "CC(O)CCCO")))
        elif True:
            d_4_guidance_ = (d_4_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "NCCOCCOCCN")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_4_guidance_)
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_5_og_: _dafny.Seq
            d_6_oi_: bool
            d_7_oc_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_5_og_ = out1_
            d_6_oi_ = out2_
            d_7_oc_ = out3_
            generated = d_5_og_
            insideConstrainedOut = d_6_oi_
            currentConstrainedOut = d_7_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    elif ((parser).IsCompletePrefix(currentConstrainedOut)) and ((len(currentConstrainedOut)) >= (2)):
                        d_8_cg_: _dafny.Seq
                        d_9_ci_: bool
                        d_10_cc_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_8_cg_ = out4_
                        d_9_ci_ = out5_
                        d_10_cc_ = out6_
                        generated = d_8_cg_
                        insideConstrainedOut = d_9_ci_
                        currentConstrainedOut = d_10_cc_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_11_constrainedPrompt_: _dafny.Seq
                        d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_12_curLen_: int
                        d_12_curLen_ = len(currentConstrainedOut)
                        d_13_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if (d_12_curLen_) < (8):
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), eosToken)
                            d_13_next_ = out7_
                        elif True:
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_13_next_ = out8_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_13_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_14_ag_: _dafny.Seq
                            d_15_ai_: bool
                            d_16_ac_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                            d_14_ag_ = out9_
                            d_15_ai_ = out10_
                            d_16_ac_ = out11_
                            generated = d_14_ag_
                            insideConstrainedOut = d_15_ai_
                            currentConstrainedOut = d_16_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_17_closeBudget_: int
            d_17_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_18_cg_: _dafny.Seq
            d_19_ci_: bool
            d_20_cc_: _dafny.Seq
            out12_: _dafny.Seq
            out13_: bool
            out14_: _dafny.Seq
            out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_17_closeBudget_)
            d_18_cg_ = out12_
            d_19_ci_ = out13_
            d_20_cc_ = out14_
            generated = d_18_cg_
            insideConstrainedOut = d_19_ci_
            currentConstrainedOut = d_20_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

