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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single valid acrylate SMILES string. Acrylates must contain vinyl ester group C=CC(=O)O-. Use diverse R-groups: ethyl (CC), propyl (CCC), butyl (CCCC), isopropyl (C(C)C), isobutyl (CC(C)C), tert-butyl (C(C)(C)C), 2-ethylhexyl (CCCCC(CC)C), hydroxyethyl (CCO), methoxyethyl (CCOC), cyclohexyl (C1CCCCC1), benzyl (Cc1ccccc1), phenyl (c1ccccc1), furfuryl (Cc1ccco1), glycidyl (CC1CO1), dimethylaminoethyl (CCN(C)C), tetrahydrofurfuryl (CC1CCCO1). Examples: C=CC(=O)OCCC, C=CC(=O)OC(C)C, C=CC(=O)OCCCC, C=CC(=O)OCC(CC)CCCC, C=CC(=O)OCCO, C=CC(=O)OCCOC, C=CC(=O)OCC1CO1, C=CC(=O)OC1CCCCC1, C=CC(=O)OCc1ccccc1. Output ONLY the SMILES with no extra text.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
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
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_5_remaining_: int
            d_5_remaining_ = (maxSteps) - (d_1_steps_)
            d_6_closeBudget_: int
            if (d_5_remaining_) < (55):
                d_6_closeBudget_ = d_5_remaining_
            elif True:
                d_6_closeBudget_ = 55
            d_7_cg_: _dafny.Seq
            d_8_ci_: bool
            d_9_cc_: _dafny.Seq
            out3_: _dafny.Seq
            out4_: bool
            out5_: _dafny.Seq
            out3_, out4_, out5_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_6_closeBudget_)
            d_7_cg_ = out3_
            d_8_ci_ = out4_
            d_9_cc_ = out5_
            generated = d_7_cg_
            insideConstrainedOut = d_8_ci_
            currentConstrainedOut = d_9_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

