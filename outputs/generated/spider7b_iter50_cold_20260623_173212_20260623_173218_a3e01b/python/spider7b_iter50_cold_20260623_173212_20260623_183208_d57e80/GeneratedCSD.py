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
        if (maxSteps) == (0):
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a single valid SQL query. Output format: SQL: YOUR QUERY. Start the query with SELECT or WITH.")))
        d_1_preambleSteps_: int
        d_1_preambleSteps_ = 0
        d_2_maxPreamble_: int
        d_2_maxPreamble_ = 5
        if (d_2_maxPreamble_) > (maxSteps):
            d_2_maxPreamble_ = maxSteps
        while ((d_1_preambleSteps_) < (d_2_maxPreamble_)) and ((cost) < (maxSteps)):
            d_3_next_: _dafny.Seq
            out0_: _dafny.Seq
            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
            d_3_next_ = out0_
            cost = (cost) + (1)
            d_1_preambleSteps_ = (d_1_preambleSteps_) + (1)
            if (d_3_next_) == (eosToken):
                return generated, insideConstrainedOut, currentConstrainedOut, cost
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
        d_4_sqlAccum_: _dafny.Seq
        d_4_sqlAccum_ = _dafny.SeqWithoutIsStrInference([])
        d_5_preambleGenerated_: _dafny.Seq
        d_5_preambleGenerated_ = generated
        d_6_constrainedPrompt_: _dafny.Seq
        d_6_constrainedPrompt_ = (prompt) + (d_5_preambleGenerated_)
        d_7_hitEos_: bool
        d_7_hitEos_ = False
        with _dafny.label("0"):
            while ((cost) < (maxSteps)) and (not(d_7_hitEos_)):
                with _dafny.c_label("0"):
                    if ((cost) + (1)) > (maxSteps):
                        raise _dafny.Break("0")
                    d_8_currentOut_: _dafny.Seq
                    d_9_eos_: bool
                    d_10_stepsUsed_: int
                    out1_: _dafny.Seq
                    out2_: bool
                    out3_: int
                    out1_, out2_, out3_ = (d_0_helpers_).ConstrainedSymbol(lm, parser, d_6_constrainedPrompt_, d_4_sqlAccum_, 1, eosToken)
                    d_8_currentOut_ = out1_
                    d_9_eos_ = out2_
                    d_10_stepsUsed_ = out3_
                    cost = (cost) + (d_10_stepsUsed_)
                    if d_9_eos_:
                        d_7_hitEos_ = True
                        d_4_sqlAccum_ = d_8_currentOut_
                        generated = (d_5_preambleGenerated_) + (d_4_sqlAccum_)
                        raise _dafny.Break("0")
                    if (len(d_8_currentOut_)) <= (len(d_4_sqlAccum_)):
                        raise _dafny.Break("0")
                    d_4_sqlAccum_ = d_8_currentOut_
                    generated = (d_5_preambleGenerated_) + (d_4_sqlAccum_)
                    if (parser).IsCompletePrefix(d_4_sqlAccum_):
                        raise _dafny.Break("0")
                    pass
            pass
        insideConstrainedOut = False
        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        return generated, insideConstrainedOut, currentConstrainedOut, cost

