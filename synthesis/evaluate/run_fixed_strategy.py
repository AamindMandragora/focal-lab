"""Compatibility entrypoint for fixed strategies.

This module now delegates to the legacy-backed fixed-strategy runner.
"""

from synthesis.evaluate.run_legacy_fixed_strategy import main


if __name__ == "__main__":
    main()
