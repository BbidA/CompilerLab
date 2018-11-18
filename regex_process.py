from functools import reduce

CAT = '•'

_priority_table = {CAT: 0, '|': 1, '*': 2, '?': 2}

_operators = (CAT, '|', '*', '?', '(', ')')

_escaped_characters = ('\t', '\n')


def is_action(character: str):
    return character not in _operators


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
    for i in range(len(regular_expr)):
        ch = regular_expr[i]
        if i > 1 and regular_expr[i - 1] == '\\':
            postfix += ch
            continue
        elif ch == '\\':
            continue

        if is_action(ch):
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
            if ch == '*' or ch == '?':
                # star operator just append to the postfix, no need
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

        if curr_char == '(' or curr_char == '|' or curr_char == '\\':
            continue
        elif is_action(next_char) or next_char == '(':
            result += CAT
    result += regular_expr[-1]
    return result


def _process_square_brackets_expr(regular_expr):
    matches = _find_all_square_brackets(regular_expr)
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


def _find_all_square_brackets(regular_expr: str):
    result = []
    for i in range(len(regular_expr) - 1):
        # process escape character
        if (regular_expr[i] == '[' and i > 0 and regular_expr[i - 1] != '\\') or (i == 0 and regular_expr[i] == '['):
            pointer = i + 1
            while pointer < len(regular_expr) and (regular_expr[pointer] != ']' or regular_expr[pointer - 1] == '\\'):
                pointer += 1
            if pointer == len(regular_expr):
                raise ValueError("Invalid [ at {}".format(i))
            else:
                result.append(regular_expr[i:pointer + 1])

    return result


def _transform_dot_sign(regular_expr: str):
    dot_replacement = _process_square_brackets_expr('[ -~]')
    result = ''
    if regular_expr[0] == '.':
        result += dot_replacement
    for i in range(1, len(regular_expr)):
        if regular_expr[i] == '.' and regular_expr[i - 1] != '\\':
            result += dot_replacement
        else:
            result += regular_expr[i]

    return result


def _transform_plus_sign(regular_expr):
    if regular_expr[0] == '+':
        raise ValueError("+ is invalid at position 0")

    result = ''
    for i in range(1, len(regular_expr)):
        if regular_expr[i] == '+' and regular_expr[i - 1] != '\\':
            prev_ch = regular_expr[i - 1]
            if prev_ch == ')':
                pointer = i - 1
                while pointer >= 0 and regular_expr[pointer] != '(':
                    pointer -= 1
                if pointer < 0:
                    raise ValueError(") is invalid in position {}".format(i))
        else:
            result += regular_expr[i]
    return regular_expr


def _transform_question_mark(regular_expr):
    # todo
    return regular_expr


def _replace_brace_content(regular_expr):
    # todo
    return regular_expr
