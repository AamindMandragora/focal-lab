"""SMILES validity, uniqueness, and target-class membership metrics."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Iterable, Sequence

from lark import Lark
from lark.exceptions import LarkError

try:  # Optional on focal; metrics report when it is absent.
    from rdkit import Chem  # type: ignore
except Exception:  # pragma: no cover - depends on host environment
    Chem = None  # type: ignore

EOS_MARKERS = ("<|im_end|>", "<|eot_id|>", "<|endoftext|>")
CLASS_MOTIFS: dict[str, tuple[str, ...]] = {
    "acrylates": (
        "C=CC(=O)O",
        "C=CC(O)=O",
        "C(=C)C(=O)O",
        "C(=C)C(O)=O",
        "OC(=O)C=C",
        "O=C(O)C=C",
        "OC(=O)C(=C)",
        "O=C(O)C(=C)",
    ),
    "chain_extenders": ("OC", "CO", "N"),
    "isocyanates": ("N=C=O", "O=C=N"),
}


def clean_smiles_output(output: str | None) -> str:
    if output is None:
        return ""
    text = str(output).strip()
    for marker in EOS_MARKERS:
        text = text.replace(marker, "")
    text = text.replace("<<", "").replace(">>", "")
    text = text.split("\n\n", 1)[0].strip()
    if "Molecule:" in text:
        text = text.rsplit("Molecule:", 1)[-1].strip()
    return text.splitlines()[0].strip() if text else ""


@lru_cache(maxsize=None)
def _lark_parser(grammar_text: str) -> Lark:
    return Lark(grammar_text, start="start", parser="lalr")


def grammar_valid(smiles: str, grammar_text: str) -> bool:
    if not smiles:
        return False
    try:
        _lark_parser(grammar_text).parse(smiles)
        return True
    except LarkError:
        return False


def rdkit_available() -> bool:
    return Chem is not None


def rdkit_valid(smiles: str) -> bool | None:
    if Chem is None:
        return None
    if not smiles:
        return False
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def target_class_membership(class_name: str, smiles: str) -> bool:
    motifs = CLASS_MOTIFS.get(class_name)
    if not motifs or not smiles:
        return False
    return any(motif in smiles for motif in motifs)


def is_prompt_exemplar(smiles: str, prompt_exemplars: Sequence[str]) -> bool:
    return smiles in set(prompt_exemplars)


def evaluate_smiles_output(
    class_name: str,
    output: str | None,
    grammar_text: str,
    prompt_exemplars: Sequence[str],
) -> Dict[str, Any]:
    smiles = clean_smiles_output(output)
    grammar_ok = grammar_valid(smiles, grammar_text)
    rdkit_ok = rdkit_valid(smiles)
    syntax_ok = grammar_ok and (rdkit_ok if rdkit_ok is not None else True)
    membership_ok = target_class_membership(class_name, smiles)
    exemplar = is_prompt_exemplar(smiles, prompt_exemplars)
    unique_valid_candidate = bool(smiles and syntax_ok and membership_ok and not exemplar)
    return {
        "smiles": smiles,
        "grammar_valid": grammar_ok,
        "rdkit_available": rdkit_available(),
        "rdkit_valid": rdkit_ok,
        "syntax_valid": syntax_ok,
        "class_membership": membership_ok,
        "is_prompt_exemplar": exemplar,
        "unique_valid_candidate": unique_valid_candidate,
    }
