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
        return base_grammar

    sorted_vars = sorted(variables, key=len, reverse=True)
    var_rule_body = " | ".join(f'"{v}"' for v in sorted_vars)
    new_var_rule = f'VARIABLE: {var_rule_body}'

    result = base_grammar

    # Replace explicit VARIABLE rule (e.g. VARIABLE: /regex/)
    pattern_var = r'^VARIABLE:.*$'
    if re.search(pattern_var, result, re.MULTILINE):
        result = re.sub(pattern_var, new_var_rule, result, flags=re.MULTILINE)
    else:
        # Replace %import common.CNAME -> VARIABLE with an explicit rule
        import_pattern = r'^%import\s+common\.CNAME\s*->\s*VARIABLE\s*$'
        if re.search(import_pattern, result, re.MULTILINE):
            result = re.sub(import_pattern, new_var_rule, result, flags=re.MULTILINE)
        else:
            result = result + "\n" + new_var_rule

    return result


def build_numeric_only_grammar(base_grammar: str) -> str:
    """Build a GSM grammar variant that accepts numeric expressions only."""
    return re.sub(
        r'\?any_expr: s_expr\s*\n\s*\|\s*n_expr',
        '?any_expr: n_expr',
        base_grammar,
        count=1,
    )


def crane_valid_vars_gsm_grammar(valid_vars: List[str]) -> str:
    """Per-example GSM grammar matching CRANE ``get_valid_vars_gsm_grammar``.

    Used for constrained decode (adaptive CRANE parity). Scoring syntax still uses
    the static ``gsm.lark`` file via ``eval_logic._final_block_parser``.
    """
    if not valid_vars:
        raise ValueError("crane_valid_vars_gsm_grammar requires at least one variable")
    var_production = " | ".join(f'"{var}"' for var in valid_vars)
    return f"""start: space? "<" "<" space? expr space? ">" ">" space?

expr: expr space? "+" space? term
     | expr space? "-" space? term
     | term

term: term space? "*" space? factor
     | term space? "/" space? factor
     | term space? "//" space? factor
     | term space? "%" space? factor
     | factor space?

factor: "-" space? factor
       | TYPE "(" space? expr space? ")"
       | primary space?

primary: NUMBER
        | var
        | "(" space? expr space? ")"

var: {var_production}

TYPE.4: "int"

space: " "

%import common.CNAME -> VARIABLE
%import common.NUMBER"""


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
