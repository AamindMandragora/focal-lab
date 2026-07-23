"""Domain / sibling helpers for the 20-cell inventory."""

from __future__ import annotations

MODELS = ("qwen25-1p5b", "qwen25-7b", "qwen35-2b", "qwen35-4b")
DOMAINS = (
    "gsm",
    "spider",
    "smiles-acrylates",
    "smiles-chain_extenders",
    "smiles-isocyanates",
)


def domain_key(cell_id: str) -> str:
    """Return gsm | spider | smiles-<class> for a cell id."""
    for domain in DOMAINS:
        prefix = f"{domain}-"
        if cell_id.startswith(prefix):
            return domain
    raise ValueError(f"unknown cell domain for {cell_id!r}")


def lane_name(cell_id: str) -> str:
    key = domain_key(cell_id)
    if key.startswith("smiles-"):
        return "smiles"
    return key


def same_domain_siblings(cell_id: str, all_cell_ids: list[str]) -> list[str]:
    key = domain_key(cell_id)
    return [c for c in all_cell_ids if c != cell_id and domain_key(c) == key]


def build_20_cell_ids() -> list[str]:
    return [f"{domain}-{model}" for domain in DOMAINS for model in MODELS]
