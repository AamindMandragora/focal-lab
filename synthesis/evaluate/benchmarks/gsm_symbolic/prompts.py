from __future__ import annotations


def reasoning_with_symbolic_expr_prompt(question: str) -> str:
    return (
        "You are an expert in solving grade school math tasks. "
        "You will be presented with a grade-school math word problem with symbolic variables and be asked to solve it.\n\n"
        "Before answering you should reason about the problem (using the <reasoning> field in the response described below). "
        "Intermediate symbolic expressions generated during reasoning should be wrapped in << >>.\n\n"
        "Then, output the symbolic expression wrapped in << >> that answers the question. "
        "The expressions must use numbers as well as the variables defined in the question. "
        "You are only allowed to use the following operations: +, -, /, //, %, (), and int().\n\n"
        "You will always respond in the format described below:\n"
        "Let's think step by step. <reasoning> The final answer is <<symbolic expression>>\n\n"
        f"Problem: {question}\n"
        "Response:\n"
    )
