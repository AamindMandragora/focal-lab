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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one valid SQLite query using only the provided schema context. The decoder will prefill SQL: <<, so continue directly with the SQL query, usually starting with SELECT or WITH. Do not repeat SQL:, do not explain, do not use Markdown, and do not add text after the query. Prefer the simplest semantically correct query with exact table and column names."))
        if (len(validTokenGroups)) > (0):
            d_1_guidance_ = (d_1_guidance_) + (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " Treat contextual token groups as schema/name hints when relevant.")))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_headerStage_: int
        d_2_headerStage_ = 0
        d_3_steps_: int
        d_3_steps_ = 0
        with _dafny.label("0"):
            while (d_3_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if insideConstrainedOut:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_4_closedGenerated_: _dafny.Seq
                            d_5_closedInside_: bool
                            d_6_closedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_4_closedGenerated_ = out0_
                            d_5_closedInside_ = out1_
                            d_6_closedCurrent_ = out2_
                            generated = d_4_closedGenerated_
                            insideConstrainedOut = d_5_closedInside_
                            currentConstrainedOut = d_6_closedCurrent_
                            d_3_steps_ = (d_3_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_7_constrainedPrompt_: _dafny.Seq
                            d_7_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_8_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_8_next_ = out3_
                            d_3_steps_ = (d_3_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_9_valid_: bool
                                out4_: bool
                                out4_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_8_next_)
                                d_9_valid_ = out4_
                                if d_9_valid_:
                                    d_10_appendedGenerated_: _dafny.Seq
                                    d_11_appendedInside_: bool
                                    d_12_appendedCurrent_: _dafny.Seq
                                    out5_: _dafny.Seq
                                    out6_: bool
                                    out7_: _dafny.Seq
                                    out5_, out6_, out7_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_8_next_)
                                    d_10_appendedGenerated_ = out5_
                                    d_11_appendedInside_ = out6_
                                    d_12_appendedCurrent_ = out7_
                                    generated = d_10_appendedGenerated_
                                    insideConstrainedOut = d_11_appendedInside_
                                    currentConstrainedOut = d_12_appendedCurrent_
                    elif (d_2_headerStage_) == (0):
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:"))]))
                        d_2_headerStage_ = 1
                        d_3_steps_ = (d_3_steps_) + (1)
                    elif (d_2_headerStage_) == (1):
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " "))]))
                        d_2_headerStage_ = 2
                        d_3_steps_ = (d_3_steps_) + (1)
                    elif (d_2_headerStage_) == (2):
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
                        d_2_headerStage_ = 3
                        d_3_steps_ = (d_3_steps_) + (1)
                    elif (d_2_headerStage_) == (3):
                        if ((d_3_steps_) + (1)) < (maxSteps):
                            d_13_chunkBudget_: int
                            d_13_chunkBudget_ = ((maxSteps) - (d_3_steps_)) - (1)
                            if (d_13_chunkBudget_) > (96):
                                d_13_chunkBudget_ = 96
                            d_14_chunkGenerated_: _dafny.Seq
                            d_15_stoppedOnOpenSpan_: bool
                            d_16_stoppedOnEos_: bool
                            d_17_chunkSteps_: int
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: bool
                            out11_: int
                            out8_, out9_, out10_, out11_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_13_chunkBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
                            d_14_chunkGenerated_ = out8_
                            d_15_stoppedOnOpenSpan_ = out9_
                            d_16_stoppedOnEos_ = out10_
                            d_17_chunkSteps_ = out11_
                            generated = d_14_chunkGenerated_
                            d_3_steps_ = (d_3_steps_) + (d_17_chunkSteps_)
                            d_2_headerStage_ = 4
                        elif True:
                            d_2_headerStage_ = 4
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                        d_3_steps_ = (d_3_steps_) + (1)
                        raise _dafny.Break("0")
                    pass
            pass
        cost = d_3_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

