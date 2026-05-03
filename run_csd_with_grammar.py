#!/usr/bin/env python3
"""Compatibility wrapper for `scripts/run_csd_with_grammar.py`."""

from scripts.run_csd_with_grammar import create_vocabulary, main, run_csd_with_grammar

__all__ = ["create_vocabulary", "main", "run_csd_with_grammar"]


if __name__ == "__main__":
    main()
