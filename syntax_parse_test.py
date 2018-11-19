import os.path

import my_lex
from syntax_algorithms import Production, Grammar, ParseTable, EPSILON

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
    '<': '<',
    '<=': '<=',
    '==': '==',
    '!=': '!=',
    '>': '>',
    '>=': '>=',
    '=': '=',
    ';': ';'
}

# construct lex
lex = my_lex.Lex(regex_token_dict.keys(), regex_token_dict)

# ---------------------------------------------------------
# read file and get tokens
curr_dir = os.path.abspath(os.path.dirname(__file__))
test_file = os.path.join(curr_dir, 'test2.txt')

with open(test_file) as f:
    characters = ''.join(f.readlines())
    result_tokens = list(filter(lambda x: x != '', lex.parse(characters)))

productions = [
    Production('S', 'id = E ; S'),
    Production('S', 'if ( C ) { S } else { S }'),
    Production('S', EPSILON),
    Production('S', 'while ( C ) { S } S'),
    Production('E', 'T E\''),
    Production('E\'', 'G E\''),
    Production('E\'', EPSILON),
    Production('G', '+ T'),
    Production('G', '- T'),
    Production('T', 'F T\''),
    Production('T\'', 'J T\''),
    Production('T\'', EPSILON),
    Production('J', '* F'),
    Production('J', '/ F'),
    Production('F', 'num'),
    Production('F', 'id'),
    Production('F', '( E )'),
    Production('C', 'D C\''),
    Production('C\'', 'or D C\''),
    Production('C\'', EPSILON),
    Production('D', 'H D\''),
    Production('D\'', 'and H D\''),
    Production('D\'', EPSILON),
    Production('H', '( C )'),
    Production('H', 'K R K'),
    Production('K', 'num'),
    Production('K', 'id'),
    Production('R', '>'),
    Production('R', '>='),
    Production('R', '<'),
    Production('R', '<='),
    Production('R', '=='),
    Production('R', '!=')
]

grammar = Grammar(productions, 'S')

parse_table = ParseTable(grammar)

parse_table.add_item(2, 'if')
parse_table.add_item(4, 'while')
parse_table.add_item(3, '}')
parse_table.add_item(1, 'id')
parse_table.add_item(3, '$')
parse_table.add_item(18, '(')
parse_table.add_item(18, 'num')
parse_table.add_item(18, 'id')
parse_table.add_item(20, ')')
parse_table.add_item(19, 'or')
parse_table.add_item(21, '(')
parse_table.add_item(21, 'num')
parse_table.add_item(21, 'id')
parse_table.add_item(23, ')')
parse_table.add_item(23, 'or')
parse_table.add_item(22, 'and')
parse_table.add_item(5, '(')
parse_table.add_item(5, 'num')
parse_table.add_item(5, 'id')
parse_table.add_item(7, ')')
parse_table.add_item(7, ';')
parse_table.add_item(6, '+')
parse_table.add_item(6, '-')
parse_table.add_item(17, '(')
parse_table.add_item(15, 'num')
parse_table.add_item(16, 'id')
parse_table.add_item(8, '+')
parse_table.add_item(9, '-')
parse_table.add_item(24, '(')
parse_table.add_item(25, 'num')
parse_table.add_item(25, 'id')
parse_table.add_item(13, '*')
parse_table.add_item(14, '/')
parse_table.add_item(26, 'num')
parse_table.add_item(27, 'id')
parse_table.add_item(10, '(')
parse_table.add_item(10, 'num')
parse_table.add_item(10, 'id')
parse_table.add_item(12, ';')
parse_table.add_item(12, '+')
parse_table.add_item(12, ')')
parse_table.add_item(12, '-')
parse_table.add_item(11, '*')
parse_table.add_item(11, '/')
parse_table.add_item(28, '>')
parse_table.add_item(29, '>=')
parse_table.add_item(30, '<')
parse_table.add_item(31, '<=')
parse_table.add_item(32, '==')
parse_table.add_item(33, '!=')

output_file = os.path.join(curr_dir, 'test2_output.txt')

with open(output_file, 'w') as f:
    i = 1
    for token in parse_table.parse(result_tokens):
        f.write('{:>2d}: {}\n'.format(i, token))
        i += 1
