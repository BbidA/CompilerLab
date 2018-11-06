import syntax_algorithms


productions = [
    syntax_algorithms.Production('E', ('T', 'E_1')),
    syntax_algorithms.Production('E_1', ('+', 'T', 'E_1')),
    syntax_algorithms.Production('E_1', None),
    syntax_algorithms.Production('T', ('F', 'T_1')),
    syntax_algorithms.Production('T_1', ('*', 'F', 'T_1')),
    syntax_algorithms.Production('T_1', None),
    syntax_algorithms.Production('F', ('(', 'E', ')', 'id'))  # need to consider how to represent id
]