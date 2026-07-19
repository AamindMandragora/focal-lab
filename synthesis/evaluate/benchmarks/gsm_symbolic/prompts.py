from __future__ import annotations


_GSM_FEWSHOTS = [
    (
        "There are {t} trees in the {g}. {g} workers will plant trees in the {g} today. "
        "After they are done, there will be {tf} trees. How many trees did the {g} workers plant today?",
        "Let's think step by step. Initially, there are {t} trees. After planting, there are "
        "{tf} trees. The number of trees planted is <<tf - t>>. The final answer is <<tf - t>>.",
    ),
    (
        "If there are {c} cars in the parking lot and {nc} more cars arrive, how many cars "
        "are in the parking lot?",
        "Let's think step by step. Initially, there are {c} cars. {nc} more cars arrive, so "
        "the total becomes <<c + nc>>. The final answer is <<c + nc>>.",
    ),
    (
        "{p1} had {ch1} {o1} and {p2} had {ch2} {o1}. If they ate {a} {o1}, how many pieces "
        "do they have left in total?",
        "Let's think step by step. Initially, {p1} had {ch1} {o1}, and {p2} had {ch2} {o1}, "
        "making a total of <<ch1 + ch2>>. After eating {a} {o1}, the remaining total is "
        "<<ch1 + ch2 - a>>. The final answer is <<ch1 + ch2 - a>>.",
    ),
    (
        "{p1} had {l1} {o1}. {p1} gave {g} {o1} to {p2}. How many {o1} does {p1} have left?",
        "Let's think step by step. {p1} started with {l1} {o1}. After giving {g} {o1} to "
        "{p2}, {p1} has <<l1 - g>> {o1} left. The final answer is <<l1 - g>>.",
    ),
    (
        "{p1} has {t} {o1}. For Christmas, {p1} got {tm} {o1} from {p2} and {td} {o1} from "
        "{p3}. How many {o1} does {p1} have now?",
        "Let's think step by step. {p1} started with {t} {o1}. {p1} received {tm} {o1} from "
        "{p2} and {td} {o1} from {p3}. The total is <<t + tm + td>>. The final answer is "
        "<<t + tm + td>>.",
    ),
    (
        "There were {c} {o1} in the server room. {nc} more {o1} were installed each day, from "
        "{d1} to {d2}. How many {o1} are now in the server room?",
        "Let's think step by step. Initially, there were {c} {o1}. {nc} {o1} were added each "
        "day for <<d2 - d1 + 1>> days, which is <<nc * (d2 - d1 + 1)>>. The total is "
        "<<c + nc * (d2 - d1 + 1)>>. The final answer is <<c + nc * (d2 - d1 + 1)>>.",
    ),
    (
        "{p1} had {gb1} {o1}. On {day1}, {p1} lost {l1} {o1}. On {day2}, {p1} lost {l2} more. "
        "How many {o1} does {p1} have at the end of {day2}?",
        "Let's think step by step. Initially, {p1} had {gb1} {o1}. After losing {l1} {o1} on "
        "{day1}, {p1} had <<gb1 - l1>>. After losing {l2} {o1} on {day2}, the total is "
        "<<gb1 - l1 - l2>>. The final answer is <<gb1 - l1 - l2>>.",
    ),
    (
        "{p1} has ${m}. {p1} bought {q} {o1} for ${p} each. How much money does {p1} have left?",
        "Let's think step by step. Initially, {p1} had ${m}. {p1} spent <<q * p>> on {q} {o1}. "
        "The remaining money is <<m - q * p>>. The final answer is <<m - q * p>>.",
    ),
]


_GSM_STD_FEWSHOTS = [
    (
        "There are {t} trees in the {g}. {g} workers will plant trees in the {g} today. "
        "After they are done, there will be {tf} trees. How many trees did the {g} workers plant today?",
        "<<tf - t>>",
    ),
    (
        "If there are {c} cars in the parking lot and {nc} more cars arrive, how many cars "
        "are in the parking lot?",
        "<<c + nc>>",
    ),
    (
        "{p1} had {ch1} {o1} and {p2} had {ch2} {o1}. If they ate {a} {o1}, how many pieces "
        "do they have left in total?",
        "<<ch1 + ch2 - a>>",
    ),
    (
        "{p1} had {l1} {o1}. {p1} gave {g} {o1} to {p2}. How many {o1} does {p1} have left?",
        "<<l1 - g>>",
    ),
    (
        "{p1} has {t} {o1}. For Christmas, {p1} got {tm} {o1} from {p2} and {td} {o1} from "
        "{p3}. How many {o1} does {p1} have now?",
        "<<t + tm + td>>",
    ),
    (
        "There were {c} {o1} in the {loc}. {nc} more {o1} were installed each day, from "
        "{d1} to {d2}. How many {o1} are now in the {loc}?",
        "<<c + nc * (d2 - d1 + 1)>>",
    ),
    (
        "{p1} had {gb1} {o1}. On {day1}, {p1} lost {l1} {o1}. On {day2}, {p1} lost {l2} more. "
        "How many {o1} does {p1} have at the end of {day2}?",
        "<<gb1 - l1 - l2>>",
    ),
    (
        "{p1} has ${m}. {p1} bought {q} {o1} for ${p} each. How much money does {p1} have left?",
        "<<m - q * p>>",
    ),
]


GSM_CRANE_COT_TASK = (
    "You are an expert in solving grade school math tasks. "
    "You will be presented with a grade-school math word problem with symbolic variables and be asked to solve it.\n\n"
    "Before answering you should reason about the problem (using the <reasoning> field in the response described below). "
    "Intermediate symbolic expressions generated during reasoning should be wrapped in << >>.\n\n"
    "Then, output the symbolic expression wrapped in << >> that answers the question. "
    "The expressions must use numbers as well as the variables defined in the question. "
    "You are only allowed to use the following operations: +, -, /, //, %, (), and int().\n\n"
    # Trailing space before \n is deliberate: CRANE's gsm_symbolic.yaml has
    # "described below: \n" and greedy Qwen3.5-2B output diverges without it
    # (verified 2026-07-02: with the space, unconstrained greedy reproduces the
    # original CRANE response 771/784 chars on eval-ex0; without it, diverges
    # at char 156). Byte-identical prompts are required for baseline parity.
    "You will always respond in the format described below: \n"
    "Let's think step by step. <reasoning> The final answer is <<symbolic expression>>"
)


def reasoning_with_symbolic_expr_prompt(question: str) -> str:
    parts = [GSM_CRANE_COT_TASK + "\n"]
    for q, a in _GSM_FEWSHOTS:
        parts.append(f"\n{q}\n\n{a}\n")
    parts.append(f"\n{question}\n")
    return "".join(parts)


def reasoning_with_symbolic_expr_messages(question: str) -> list[dict]:
    """Multi-turn chat delivery of the same few-shot reasoning prompt.

    CRANE delivers the 8 few-shot examples as alternating user/assistant turns
    (not one flattened user message). Instruct-tuned models follow the format
    far more reliably this way: on GSM-1.5B unconstrained this lifted accuracy
    22.0% -> 30.0% with zero content change. Used by eval_logic.format_prompt.
    """
    messages = [{"role": "system", "content": GSM_CRANE_COT_TASK}]
    for q, a in _GSM_FEWSHOTS:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": question})
    return messages


def symbolic_expression_only_prompt(question: str) -> str:
    """Prompt for constrained decoders: single final expression, no chain-of-thought."""
    header = (
        "You are an expert in solving grade school math tasks. "
        "You will be presented with a grade-school math word problem with symbolic variables and be asked to solve it.\n\n"
        "Only output the symbolic expression wrapped in << >> that answers the question. "
        "The expression must use numbers as well as the variables defined in the question. "
        "You are only allowed to use the following operations: +, -, /, //, %, (), and int().\n\n"
        "You will always respond in the format described below:\n"
        "<<symbolic expression>>\n"
    )
    parts = [header]
    for q, a in _GSM_STD_FEWSHOTS:
        parts.append(f"\n{q}\n\n{a}\n")
    parts.append(f"\n{question}\n")
    return "".join(parts)
