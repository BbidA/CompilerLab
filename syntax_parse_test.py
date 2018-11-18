import my_lex
import os.path

from base import Token

# -------------------------------------------
# define lex rules to produce lex tokens

# regex definition
delimiter = '[ \t\n]'
ws = '{}{}*'.format(delimiter, delimiter)
letter = '[A-Za-z]'
digit = '[0-9]'
id_token = '(_|{})(_|{}|{})*'.format(letter, letter, digit)
number = '{}{}*(.{}{}*)?(E(+|-)?{}{}*)?'.format(digit, digit, digit, digit, digit, digit)

# regex-token definition
regex_token_dict = {
    ws: '',
    'if': 'if',
    'else': 'else',
    'while': 'while',
    'or': 'or',
    'and': 'and',
    '+': '+',
    '-': '-',
    '/': '/',
    '*': '*',
    '(': '(',
    ')': ')',
    '{': '{',
    '}': '}',
    id_token: 'id',
    number: 'num',
    '<': 'COP',
    '<=': 'COP',
    '==': 'COP',
    '!=': 'COP',
    '>': 'COP',
    '>=': 'COP',
    '=': '=',
    ';': ';'
}

# construct lex
lex = my_lex.Lex(regex_token_dict.keys(), regex_token_dict)

# ---------------------------------------------------------
# read file and get tokens
curr_dir = os.path.abspath(os.path.dirname(__file__))
test_file = os.path.join(curr_dir, 'test2.txt')
print(test_file)

with open(test_file) as f:
    characters = ''.join(f.readlines())
    result_tokens = list(filter(lambda x: x != '', lex.parse(characters)))
    print(result_tokens)
