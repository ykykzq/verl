# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Conservative answer normalization shared by DeepScaleR ingestion and scoring."""

import re
from fractions import Fraction

from pylatexenc.latexwalker import (
    LatexCharsNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexMathNode,
    LatexWalker,
    LatexWalkerError,
    get_default_latex_context_db,
)
from pylatexenc.macrospec import MacroSpec

from .math_dapo import last_boxed_only_string, remove_boxed

NORMALIZATION_VERSION = "deepscaler-format-v1"
_CONTEXT = get_default_latex_context_db()
_CONTEXT.add_context_category(
    "deepscaler", macros=[MacroSpec("dfrac", "{{"), MacroSpec("tfrac", "{{"), MacroSpec("boxed", "{")], prepend=True
)
_NUMBER = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
_FRACTION = re.compile(r"\\frac\{(" + _NUMBER + r")\}\{(" + _NUMBER + r")\}")
_SLASH_FRACTION = re.compile(r"([+-]?[0-9]+)/([+-]?[0-9]+)")
_BOX = re.compile(r"(?<!\\)\\boxed(?![A-Za-z])")


def _walker(text):
    return LatexWalker(text, latex_context=_CONTEXT, tolerant_parsing=False)


def _render(nodes):
    return "".join(_render_node(node) for node in nodes)


def _argument(node):
    if isinstance(node, LatexMacroNode) and node.nodeargd is None:
        # A bare control sequence argument has no parsed args; braces now delimit
        # its name, so its original control-word separator is unnecessary.
        return "\\" + node.macroname
    return _render(node.nodelist) if isinstance(node, LatexGroupNode) else _render_node(node)


def _render_node(node):
    if isinstance(node, LatexMacroNode) and node.nodeargd is not None:
        args = node.nodeargd.argnlist
        if node.macroname in ("frac", "dfrac", "tfrac"):
            return "\\frac{" + _argument(args[0]) + "}{" + _argument(args[1]) + "}"
        if node.macroname == "sqrt":
            index = "[" + _argument(args[0]) + "]" if args[0] is not None else ""
            return "\\sqrt" + index + "{" + _argument(args[1]) + "}"
    if isinstance(node, LatexGroupNode):
        return node.delimiters[0] + _render(node.nodelist) + node.delimiters[1]
    # Preserve text, unknown macros, internal math delimiters and whitespace.
    return node.latex_verbatim()


def normalize_answer(answer: str) -> str:
    """Normalize supported LaTeX formatting and whole exact rational numbers.

    This is not a symbolic equivalence checker. Unrecognized or malformed syntax
    remains literal; units, commas, internal whitespace and text are not removed.
    """
    answer = answer.strip()
    try:
        nodes, _, _ = _walker(answer).get_latex_nodes()
        while True:
            while nodes and isinstance(nodes[0], LatexCharsNode) and not nodes[0].chars.strip():
                nodes = nodes[1:]
            while nodes and isinstance(nodes[-1], LatexCharsNode) and not nodes[-1].chars.strip():
                nodes = nodes[:-1]
            if len(nodes) != 1:
                break
            node = nodes[0]
            if isinstance(node, LatexMathNode):
                nodes = node.nodelist
            elif isinstance(node, LatexMacroNode) and node.macroname == "boxed":
                argument = node.nodeargd.argnlist[0]
                if not isinstance(argument, LatexGroupNode):
                    break
                nodes = argument.nodelist
            else:
                break
        normalized = _render(nodes).strip()
    except (LatexWalkerError, RecursionError):
        return answer

    try:
        if re.fullmatch(_NUMBER, normalized):
            rational = Fraction(normalized)
        elif match := _FRACTION.fullmatch(normalized) or _SLASH_FRACTION.fullmatch(normalized):
            rational = Fraction(match[1]) / Fraction(match[2])
        else:
            return normalized
        if rational.denominator == 1:
            return str(rational.numerator)
        return f"\\frac{{{rational.numerator}}}{{{rational.denominator}}}"
    except (ValueError, ZeroDivisionError):
        return normalized


def compute_score(solution_str: str, ground_truth: str) -> float:
    """Compare only the final complete boxed answer, with the ingestion contract."""
    last_box = None
    for last_box in _BOX.finditer(solution_str):
        pass
    if last_box is None:
        return 0.0
    suffix = solution_str[last_box.end() :].lstrip()
    if not suffix.startswith("{"):
        return 0.0
    # Reuse the boxed-span extractor without requiring the answer itself to be
    # valid LaTeX (e.g. a literal currency amount "$87.50").
    boxed = last_boxed_only_string("\\boxed" + suffix)
    if boxed is None:
        return 0.0
    answer = normalize_answer(remove_boxed(boxed))
    return float(bool(answer) and answer == normalize_answer(ground_truth))
