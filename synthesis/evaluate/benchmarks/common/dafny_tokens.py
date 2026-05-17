"""Shared helpers for Dafny token/sequence conversion in evaluation runtimes."""

from __future__ import annotations


def dafny_seq_to_str(seq) -> str:
    """
    Convert a Dafny Seq to a Python string.

    Dafny.Seq objects have __len__ and __getitem__ but NOT __iter__,
    so ''.join(seq) fails. Use index-based iteration as a fallback.
    """
    try:
        return "".join(seq)
    except TypeError:
        try:
            return "".join(seq[i] for i in range(len(seq)))
        except (TypeError, AttributeError, IndexError):
            return str(seq)
