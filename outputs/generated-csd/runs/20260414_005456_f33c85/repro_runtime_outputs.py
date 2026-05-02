import sys
from pathlib import Path

sys.path.insert(0, "/home/aadivyar/csd-generation")

from synthesis.runner import StrategyRunner


def main() -> None:
    base = Path(__file__).resolve().parent
    names = [
        "gsm_gpt54_phase_switching_40acc_v9",
        "gsm_gpt54_phase_switching_40acc_v9_20260414_005658_015492",
        "gsm_gpt54_phase_switching_40acc_v9_20260414_005746_07db3a",
    ]

    for name in names:
        runner = StrategyRunner(max_steps=100, parser_mode="permissive")
        result = runner.run(base / name / "GeneratedCSD.py")
        print("DIR", name)
        print("SUCCESS", result.success)
        print("ERROR_TYPE", result.error_type)
        print("ERROR_MESSAGE", result.error_message)
        print("COST", result.cost)
        print("OUT", result.output)
        print("TEXT", "".join(result.output or []))
        print("---")


if __name__ == "__main__":
    main()
