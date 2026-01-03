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
    def GeneratedStrategy():
        return VerifiedDecoderAgent.Strategy_TryK(3, VerifiedDecoderAgent.Attempt_Unconstrained(), VerifiedDecoderAgent.CheckPredicate_ParseOk(), VerifiedDecoderAgent.Strategy_Constrained(VerifiedDecoderAgent.TokenConstraint_GrammarMask()))

