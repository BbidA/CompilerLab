# Synopsis:
# RE -> DFA
# Integrate multi DFAs to a NFA
# Transform NFA to DFA
import re

from functools import reduce

_priority_table = {'•': 0, '|': 1, '*': 2, '+': 2, '?': 2}

_unary_operator = ('*', '+', '?')


def to_postfix(regular_expr):
    """Transform the regular expression to postfix expression

    Parameters
    ----------
    regular_expr: str
        regular expression to be processed

    Returns
    -------
    postfix_form: str
        postfix form of this regular expression
    """

    regular_expr = _pre_process_regular_expr(regular_expr)
    stack = []
    postfix = ''
    for ch in regular_expr:
        if ch.isalpha() or ch.isdigit():
            postfix += ch
        elif ch == '(':
            stack.append(ch)
        elif ch == ')':
            while stack[-1] != '(':
                postfix += stack.pop()
            stack.pop()
        else:
            while len(stack) != 0 and stack[-1] != '(' and _priority_table[stack[-1]] >= _priority_table[ch]:
                postfix += stack.pop()
            if ch in _unary_operator:
                # unary operator just append to the postfix, no need
                # to push into the stack
                postfix += ch
            else:
                stack.append(ch)

    for rest in stack:
        postfix += rest

    return postfix


def _pre_process_regular_expr(regular_expr):
    """Pre-process a regular expression to eliminate '[]'

    Parameters
    ----------
    regular_expr: str
        regular expression to be processed

    Returns
    -------
    processed_re: str
        regular expression without operator '[]'

    """
    regular_expr = _process_square_brackets_expr(regular_expr)
    return _add_cat_operator(regular_expr)


def _add_cat_operator(regular_expr):
    result = ''
    for i in range(len(regular_expr) - 1):
        curr_char = regular_expr[i]
        result += curr_char
        next_char = regular_expr[i + 1]

        if curr_char == '(' or curr_char == '|':
            continue
        elif next_char.isalpha() or next_char.isdigit() or next_char == '(':
            result += '•'
    result += regular_expr[-1]
    return result


def _process_square_brackets_expr(regular_expr):
    matches = re.findall(r'\[.*\]', regular_expr)
    for m in matches:
        result = []
        for i in range(1, len(m) - 1):
            if m[i - 1] == '-':
                assert i - 2 > 0
                for character in range(ord(m[i - 2]) + 1, ord(m[i]) + 1):
                    result.append(chr(character))
            elif m[i] == '-':
                continue
            else:
                result.append(m[i])
        replacement = '(' + reduce(lambda a, b: a + '|' + b, result) + ')'
        regular_expr = regular_expr.replace(m, replacement)
    return regular_expr
