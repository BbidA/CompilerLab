EPSILON = 'epsilon'


class Production:

    def __init__(self, left_part, right_part):
        self.left_part = left_part
        self.right_part = right_part

    def __repr__(self):
        return '{} --> {}'.format(self.left_part, self.right_part)

    def __str__(self):
        return self.__repr__()

    @classmethod
    def epsilon_production(cls, left_part):
        return Production(left_part, EPSILON)

    @property
    def right_part_signs(self):
        return self.right_part.split(' ')


class Grammar:

    def __init__(self, productions, start_sign):
        self.productions = productions
        self.start_sign = start_sign
        self.non_final_signs = set()
        self.final_signs = set()

        self._init_signs()

    def _init_signs(self):
        all_signs = set()
        for production in self.productions:
            self.non_final_signs.add(production.left_part)
            all_signs.update(production.right_part_signs)

        self.final_signs.update(all_signs.difference(self.non_final_signs))


class ParseTable:

    def __init__(self, grammar):
        self.table = {}
        self.grammar = grammar

    def add_item(self, production_index, input_sign):
        production = self.grammar.productions[production_index - 1]
        self.table[production.left_part, input_sign] = production

    def parse(self, input_tokens):
        input_tokens = input_tokens + ['$']  # add a terminate sign

        stack = [self.grammar.start_sign]
        result = []  # result is the leftmost derivation of the input_str
        pointer = 0

        while len(stack) != 0:
            top_item = stack.pop()

            if top_item in self.grammar.final_signs:
                if top_item == input_tokens[pointer]:
                    pointer += 1
                elif top_item == EPSILON:
                    # epsilon production
                    continue
                else:
                    raise ValueError('Unmatched string')
            elif top_item in self.grammar.non_final_signs:
                next_production = self.table.get((top_item, input_tokens[pointer]))
                if next_production is None:
                    raise ValueError('Unmatched string')

                result.append(next_production)

                # push right part of the production into the stack
                for ch in next_production.right_part_signs[::-1]:
                    stack.append(ch)

        return result
