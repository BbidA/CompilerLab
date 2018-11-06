import my_lex

tokens = (
    'LT', 'LE', 'EQ', 'NE', 'GT', 'GE',
    'IF', 'THEN', 'ELSE', 'ID', 'NUMBER', 'RELOP'
)

# Tokens

delimiter = '[ \t\n]'
ws = '{}{}*'.format(delimiter, delimiter)
letter = '[A-Za-z]'
digit = '[0-9]'
t_id = '(_|{})(_|{}|{})*'.format(letter, letter, digit)
number = '{}{}*(.{}{}*)?(E(+|-)?{}{}*)?'.format(digit, digit, digit, digit, digit, digit)


class IDGenerator:

    def __init__(self):
        self._id = 0
        self._number = 0

    def next_id(self):
        curr = self._id
        self._id += 1
        return '[ID, {}]'.format(curr)

    def next_number(self):
        result = '[NUMBER, {}]'.format(self._number)
        self._number += 1
        return result


id_gen = IDGenerator()

actions = dict()
actions[ws] = ''
actions['if'] = '[IF]'
actions['then'] = '[THEN]'
actions['else'] = '[ELSE]'
actions[t_id] = id_gen.next_id
actions[number] = id_gen.next_number
actions['<'] = '[RELOP, LE]'
actions['<='] = '[RELOP, LE]'
actions['='] = '[RELOP, EQ]'
actions['<>'] = '[RELOP, NE]'
actions['>'] = '[RELOP, GT]'
actions['>='] = '[RELOP, GE]'

# construct the dfa
regular_expression = [ws, 'if', 'then', 'else', t_id, number, '<', '<=', '=', '<>', '>', '>=']
multi_nfa = [my_lex.construct_nfa(regular_expr) for regular_expr in regular_expression]
nfa = my_lex.integrate_multi_nfa(multi_nfa)
dfa = my_lex.DFA.subset_construction(nfa)

with open('test.txt', 'r') as f:
    string = ''.join(f.readlines())
    result_tokens = dfa.parse_string(string)
    with open('test_output.txt', 'w') as out:
        for t in result_tokens:
            if t == 'Invalid character':
                out.writelines(t + '\n')
            else:
                to_write = actions[t] if not callable(actions[t]) else actions[t]()
                print(to_write)
                out.writelines(to_write + '\n')
