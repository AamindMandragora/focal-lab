from syncode.grammar_decoder import SyncodeLogitsProcessor
from syncode.parsers.grammars import Grammar

# Optional CLI dependency path: infer.py imports `fire`.
# Evaluation only needs parser/mask modules, so keep package import usable even
# when `fire` is absent.
try:
    from syncode.infer import Syncode, AdaptiveSynCode
except ModuleNotFoundError:  # pragma: no cover - depends on host env
    Syncode = None
    AdaptiveSynCode = None
