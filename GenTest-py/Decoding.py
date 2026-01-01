import sys
from typing import Callable, Any, TypeVar, NamedTuple
from math import floor
from itertools import count

import module_ as module_
import _dafny as _dafny
import System_ as System_

# Module: Decoding

class default__:
    def  __init__(self):
        pass

    @staticmethod
    def ValidLogitsForPrefix(logits, prefix):
        def lambda0_(forall_var_0_):
            d_0_t_: _dafny.Seq = forall_var_0_
            return not ((d_0_t_) in (default__.Parser__AllowedNext(prefix))) or ((d_0_t_) in (logits))

        return _dafny.quantifier((default__.Parser__AllowedNext(prefix)).Elements, True, lambda0_)


class ProposalStrategy:
    @_dafny.classproperty
    def AllSingletonConstructors(cls):
        return [ProposalStrategy_ArgMax(), ProposalStrategy_Temperature(), ProposalStrategy_TopK(), ProposalStrategy_Nucleus()]
    @classmethod
    def default(cls, ):
        return lambda: ProposalStrategy_ArgMax()
    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)
    @property
    def is_ArgMax(self) -> bool:
        return isinstance(self, ProposalStrategy_ArgMax)
    @property
    def is_Temperature(self) -> bool:
        return isinstance(self, ProposalStrategy_Temperature)
    @property
    def is_TopK(self) -> bool:
        return isinstance(self, ProposalStrategy_TopK)
    @property
    def is_Nucleus(self) -> bool:
        return isinstance(self, ProposalStrategy_Nucleus)

class ProposalStrategy_ArgMax(ProposalStrategy, NamedTuple('ArgMax', [])):
    def __dafnystr__(self) -> str:
        return f'Decoding.ProposalStrategy.ArgMax'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, ProposalStrategy_ArgMax)
    def __hash__(self) -> int:
        return super().__hash__()

class ProposalStrategy_Temperature(ProposalStrategy, NamedTuple('Temperature', [])):
    def __dafnystr__(self) -> str:
        return f'Decoding.ProposalStrategy.Temperature'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, ProposalStrategy_Temperature)
    def __hash__(self) -> int:
        return super().__hash__()

class ProposalStrategy_TopK(ProposalStrategy, NamedTuple('TopK', [])):
    def __dafnystr__(self) -> str:
        return f'Decoding.ProposalStrategy.TopK'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, ProposalStrategy_TopK)
    def __hash__(self) -> int:
        return super().__hash__()

class ProposalStrategy_Nucleus(ProposalStrategy, NamedTuple('Nucleus', [])):
    def __dafnystr__(self) -> str:
        return f'Decoding.ProposalStrategy.Nucleus'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, ProposalStrategy_Nucleus)
    def __hash__(self) -> int:
        return super().__hash__()

