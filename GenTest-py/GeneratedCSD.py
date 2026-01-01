import sys
from typing import Callable, Any, TypeVar, NamedTuple
from math import floor
from itertools import count

import module_ as module_
import _dafny as _dafny
import System_ as System_
import Decoding as Decoding
import CSD as CSD

# Module: GeneratedCSD

class default__:
    def  __init__(self):
        pass

    @staticmethod
    def GeneratedProgram():
        return CSD.Program_ReturnParsed(0, CSD.Policy_MaskWithSynCode(CSD.Policy_WithRejection(CSD.Policy_Base(Decoding.ProposalStrategy_Temperature()))))

