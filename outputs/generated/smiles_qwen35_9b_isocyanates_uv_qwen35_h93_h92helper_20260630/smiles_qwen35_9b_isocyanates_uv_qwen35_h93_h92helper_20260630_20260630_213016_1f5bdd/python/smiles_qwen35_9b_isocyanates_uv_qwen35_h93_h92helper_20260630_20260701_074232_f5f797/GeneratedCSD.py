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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "OUTPUT EXACTLY ONE SMILES string for a NEW isocyanate molecule containing N=C=O. The SMILES must be RDKit-valid. Choose a UNIQUE structure not seen before. Preferred simple patterns: methylisocyanate CH3N=C=O, ethylisocyanate CCN=C=O, propylisocyanate CCCN=C=O, isopropylisocyanate CC(C)N=C=O, butylisocyanate CCCCN=C=O, tert-butylisocyanate CC(C)(C)N=C=O, cyclopropylisocyanate C1CC1N=C=O, cyclopentylisocyanate C1CCCC1N=C=O, cyclohexylisocyanate C1CCCCC1N=C=O, benzylisocyanate c1ccccc1CN=C=O, fluoromethylisocyanate FCN=C=O, chloromethylisocyanate ClCN=C=O, 2-fluoroethylisocyanate FCCN=C=O, allylisocyanate C=CCN=C=O, propargylisocyanate C#CCN=C=O. Output ONLY the SMILES, no extra text.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_2_generatedOut_: _dafny.Seq = _dafny.Seq({})
                    d_3_insideOut_: bool = False
                    d_4_currentOut_: _dafny.Seq = _dafny.Seq({})
                    d_5_done_: bool = False
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: _dafny.Seq
                    out3_: bool
                    out0_, out1_, out2_, out3_ = (d_0_helpers_).ManagedStep(lm, parser, prompt, generated, insideConstrainedOut, currentConstrainedOut, validTokenGroups, _dafny.BigRational('2e0'), 2, eosToken)
                    d_2_generatedOut_ = out0_
                    d_3_insideOut_ = out1_
                    d_4_currentOut_ = out2_
                    d_5_done_ = out3_
                    generated = d_2_generatedOut_
                    insideConstrainedOut = d_3_insideOut_
                    currentConstrainedOut = d_4_currentOut_
                    d_1_steps_ = (d_1_steps_) + (1)
                    cost = d_1_steps_
                    if d_5_done_:
                        raise _dafny.Break("0")
                    pass
            pass
        return generated, insideConstrainedOut, currentConstrainedOut, cost

