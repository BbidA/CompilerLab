class Production:

    def __init__(self, non_final_sign, right_part):
        self.non_final_sign = non_final_sign
        self.right_part = right_part


class Grammar:

    def __init__(self):
        self.productions = []
        self.non_final_signs = []
        self.final_signs = []
        self.start_sign = None


def table_driven_syntax_analysis(table, grammar, input_str):
    stack = ['S']
    result = []  # result is the leftmost derivation of the input_str
    pointer = 0

    while len(stack) != 0:
        top_item = stack.pop()

        if top_item in grammar.final_signs:
            if top_item == input_str[pointer]:
                pointer += 1
            else:
                raise ValueError('Unmatched string')
        elif top_item in grammar.non_final_signs:
            next_production = table.get((top_item, input_str[pointer]))
            if next_production is None:
                raise ValueError('Unmatched string')

            result.append(next_production)

            # this indicates it's a epsilon production
            if next_production.right_part is None:
                continue

            # push right part of the production into the stack
            for ch in next_production.right_part[::-1]:
                stack.append(ch)

    return result
