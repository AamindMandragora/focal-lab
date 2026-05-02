"""
Dynamic grammar construction for GSM-Symbolic evaluation.

Builds grammars that constrain the allowed variables based on the
specific question being evaluated.
"""

from __future__ import annotations

import re
from typing import List


def build_dynamic_grammar(base_grammar: str, variables: List[str]) -> str:
    """
    Dynamically build a grammar that ONLY allows specific variables.
    
    Replaces both the VARIABLE and VAR_LETTERS rules with explicit allowed variables.
    This prevents the model from generating expressions with arbitrary variable names.
    
    Args:
        base_grammar: The base Lark grammar string
        variables: List of allowed variable names (e.g., ["n", "m", "p"])
        
    Returns:
        Modified grammar string with constrained variable rules
        
    Example:
        >>> grammar = '''
        ... start: expr
        ... VARIABLE: /[a-z]+/
        ... '''
        >>> build_dynamic_grammar(grammar, ["x", "y"])
        '... VARIABLE: "x" | "y" ...'
    """
    if not variables:
        # Fallback if no variables found (shouldn't happen in valid GSM)
        return base_grammar

    # Sort by length descending to ensure longer vars match first (e.g. n10 before n1)
    sorted_vars = sorted(variables, key=len, reverse=True)
    var_rule_body = " | ".join(f'"{v}"' for v in sorted_vars)
    
    result = base_grammar
    
    # Replace VARIABLE rule
    new_var_rule = f'VARIABLE: {var_rule_body}'
    pattern_var = r'^VARIABLE:.*$'
    if re.search(pattern_var, result, re.MULTILINE):
        result = re.sub(pattern_var, new_var_rule, result, flags=re.MULTILINE)
    else:
        result = result + "\n" + new_var_rule
    
    # Replace VAR_LETTERS rule - must also only allow the specific variables
    # This prevents arbitrary letter sequences from being accepted
    new_var_letters_rule = f'VAR_LETTERS: {var_rule_body}'
    pattern_var_letters = r'^VAR_LETTERS:.*$'
    if re.search(pattern_var_letters, result, re.MULTILINE):
        result = re.sub(pattern_var_letters, new_var_letters_rule, result, flags=re.MULTILINE)
    
    return result


def extract_variables_from_mapping(
    variable_mapping: dict,
    include_string_variables: bool = False,
) -> List[str]:
    """
    Extract allowed variable names from a CRANE-style variable mapping.

    By default, string-valued placeholders such as person or object names are excluded
    because GSM arithmetic expressions should only reference numeric quantities.

    Args:
        variable_mapping: Dict mapping variable names to their declared types
        include_string_variables: Whether to keep entries whose type is ``"str"``

    Returns:
        List of allowed variable names
    """
    variables: List[str] = []
    for name, value_type in variable_mapping.items():
        if include_string_variables or str(value_type).lower() != 'str':
            variables.append(name)
    return variables
