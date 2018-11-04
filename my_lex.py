# Synopsis:
# RE -> DFA
# Integrate multi DFAs to a NFA
# Transform NFA to DFA
import abc
import re
from collections import deque
from functools import reduce

# ----------------------------------------------------------
# transform regular expression to its postfix form

_cat = '•'

_priority_table = {_cat: 0, '|': 1, '*': 2}

_operators = (_cat, '|', '*', '+', '?', '(', ')', '{', '}')


def _is_action(character: str):
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
            if ch == '*':
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
    regular_expr = _replace_brace_content(regular_expr)
    regular_expr = _transform_plus_sign(regular_expr)
    regular_expr = _transform_question_mark(regular_expr)
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
            result += _cat
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


def _transform_dot_sign(regular_expr):
    pass


def _transform_plus_sign(regular_expr):
    # todo
    return regular_expr


def _transform_question_mark(regular_expr):
    # todo
    return regular_expr


def _replace_brace_content(regular_expr):
    # todo
    return regular_expr


# ---------------------------------------------------------
# construct NFA

_epsilon = 'epsilon'


class NFA:

    def __init__(self):
        self.start_node = None
        self.final_nodes = []
        self.moves = {}
        self._state_count = 0
        self.nodes_count = 0

    def __getitem__(self, *state_and_action):
        return self.moves[state_and_action]

    def __str__(self):
        if self.start_node is None:
            return 'start node not specified'

        # implement a BFS to print nodes information
        result = ''
        nodes_queue = deque()
        nodes_queue.append(self.start_node)
        visited_nodes = []
        while len(nodes_queue) != 0:
            curr_state = nodes_queue.popleft()
            visited_nodes.append(curr_state)
            # this may need improved, the cost of this may be too high
            for state, action in self.moves.keys():
                if curr_state == state:
                    next_states = self.moves[state, action]
                    # add not visited state to the end of the queue
                    for s in next_states:
                        result += '{} -{}-> {}; '.format(state, action, s)
                        if s not in visited_nodes:
                            nodes_queue.append(s)
            result += '\n'
        return result

    def add_edge(self, source, action, target):
        if self.moves.get((source, action)) is None:
            self.moves[source, action] = set()

        self.moves[source, action].add(target)

    def add_epsilon_edge(self, source, target):
        self.add_edge(source, _epsilon, target)

    def new_single_action(self, action):
        head_node, last_node = self._allocate_nodes()

        self.add_edge(head_node, action, last_node)
        return head_node, last_node

    def cat(self, nodes1, nodes2):
        nodes1_tail = nodes1[1]
        nodes2_head = nodes2[0]

        # establish links between nodes1_tail and next states of nodes2_head
        to_be_popped = []
        to_be_append = {}
        for state, action in self.moves.keys():
            if nodes2_head == state:
                to_be_append[nodes1_tail, action] = self.moves[state, action]
                to_be_popped.append((state, action))
        self.moves.update(to_be_append)

        # remove nodes2_head links to other nodes
        for state_action in to_be_popped:
            self.moves.pop(state_action)
        self.nodes_count -= 1

        return nodes1[0], nodes2[1]

    def new_or_nfa(self, a_nodes, b_nodes):
        head_node, last_node = self._allocate_nodes()

        self.add_epsilon_edge(head_node, a_nodes[0])
        self.add_epsilon_edge(head_node, b_nodes[0])
        self.add_epsilon_edge(a_nodes[1], last_node)
        self.add_epsilon_edge(b_nodes[1], last_node)

        return head_node, last_node

    def new_star_nfa(self, nodes):
        head_node, last_node = self._allocate_nodes()

        self.add_epsilon_edge(head_node, nodes[0])
        self.add_epsilon_edge(nodes[1], last_node)
        self.add_epsilon_edge(nodes[1], nodes[0])
        self.add_epsilon_edge(head_node, last_node)

        return head_node, last_node

    def epsilon_closure(self, states):
        closure_set = set()
        closure_set.update(states)
        prev_closure = set()
        # use iteration strategy to find the closure
        while prev_closure != closure_set:
            prev_closure.update(closure_set)
            for (from_state, action), to_states_set in self.moves.items():
                if from_state in closure_set and action == _epsilon:
                    closure_set.update(to_states_set)

        return tuple(closure_set)

    def _allocate_nodes(self):
        head_node = self._state_count
        last_node = self._state_count + 1
        self._state_count += 2
        self.nodes_count += 2

        return head_node, last_node


def construct_nfa(regular_expr):
    """Construct NFA for a regular expression

    Parameters
    ----------
    regular_expr: str
        regular expression which is used to construct this nfa

    Returns
    -------
    nfa: dict
        target NFA which is represented by a dictionary
    """

    postfix = to_postfix(regular_expr)
    stack = []
    nfa = NFA()

    for ch in postfix:
        if _is_action(ch):
            # ch is an action
            stack.append(nfa.new_single_action(ch))
        elif ch == '*':
            assert len(stack) > 0 and isinstance(stack[-1], tuple)
            head, tail = stack.pop()
            stack.append(nfa.new_star_nfa((head, tail)))
        elif ch == '|':
            assert len(stack) >= 2
            nodes1 = stack.pop()
            nodes2 = stack.pop()
            stack.append(nfa.new_or_nfa(nodes1, nodes2))
        elif ch == _cat:
            assert len(stack) >= 2
            nodes1 = stack.pop()
            nodes2 = stack.pop()
            # sequence of nodes1 and nodes2 is important here
            stack.append(nfa.cat(nodes2, nodes1))
        else:
            raise ValueError('Illegal character: ' + ch)

    # update nfa start state and final state
    assert len(stack) == 1
    head, tail = stack.pop()
    nfa.start_node = head
    nfa.final_nodes.append(tail)

    return nfa


# ------------------------------------------------------------
# construct dfa


class DFA:

    def __init__(self):
        self.start_state = None
        self.final_states = set()
        self.nodes_count = 0
        self.moves = {}
        self.final_states_related_re = {}

        self._states_id = {}
        self._state_count = 0

    def _bond_re_to_final_states(self, state, regular_expr):
        if state not in self.final_states:
            raise ValueError("State is not a final state")

        if self.final_states_related_re.get(state) is None:
            self.final_states_related_re[state] = []

        self.final_states_related_re[state].append(regular_expr)

    def _get_id_for_states(self, states):
        # allocate an id for the states if it's not recorded yet
        # otherwise just return the recorded id
        states = tuple(states)
        allocated_id = self._states_id.get(states)

        if allocated_id is None:
            allocated_id = self._state_count
            self._state_count += 1
            self._states_id[states] = allocated_id
        return allocated_id

    def add_edge(self, source, action, target):
        if self.moves.get((source, action)) is None:
            self.moves[source, action] = set()

        self.moves[source, action].add(target)

    @classmethod
    def subset_construction(cls, nfa):
        """Use subset construction to construct a DFA from an NFA

        Parameters
        ----------
        nfa : NFA
            the NFA which is used to construct the DFA

        Returns
        -------
        dfa : DFA
            the target DFA
        """

        start_states = nfa.epsilon_closure((nfa.start_node, ))  # epsilon_closure returns a tuple of states
        state_queue = deque()  # this queue should only contains tuple of state
        state_queue.append(start_states)
        visited_states = set()

        dfa = cls()
        dfa.start_state = dfa._get_id_for_states(start_states)

        while len(state_queue) != 0:
            curr_states = state_queue.popleft()
            curr_state_id = dfa._get_id_for_states(curr_states)
            visited_states.add(curr_states)
            action_state = {}  # used to record the action and its related next states

            # find action-closure(curr_states) and save it with
            # the related action to the action_state
            for (from_state, action), to_states_set in nfa.moves.items():
                if from_state in curr_states and action != _epsilon:
                    if action_state.get(action) is None:
                        action_state[action] = set()
                    action_state[action].update(to_states_set)

            # find epsilon-closure for states in action_state
            # and add transitions to dfa
            for action, states in action_state.items():
                states.update(nfa.epsilon_closure(states))

                states_tuple = tuple(states)
                dfa.moves[curr_state_id, action] = dfa._get_id_for_states(states_tuple)
                # add not visited states to state_queue
                if states_tuple not in visited_states:
                    state_queue.append(states_tuple)
        return dfa
