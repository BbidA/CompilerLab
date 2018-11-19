# Synopsis:
# RE -> DFA
# Integrate multi DFAs to a NFA
# Transform NFA to DFA
import copy
from collections import deque

from regex_process import CAT, to_postfix, is_action

INVALID_CHARACTER = 'Invalid character'

_cat = CAT

# ---------------------------------------------------------
# construct NFA

_epsilon = 'epsilon'


class NFA:

    def __init__(self):
        self.start_node = None
        self.final_nodes = set()
        self.non_epsilon_actions = set()
        self.moves = {}
        self.final_states_related_re = {}
        self._state_count = 0
        self.nodes_count = 0

    def __str__(self):
        if self.start_node is None:
            return 'start node not specified'

        # implement a BFS to print nodes information
        result = ''
        nodes_queue = deque()
        nodes_queue.append(self.start_node)
        visited_nodes = set()
        while len(nodes_queue) != 0:
            curr_state = nodes_queue.popleft()
            visited_nodes.add(curr_state)

            # jump the final state
            if curr_state in self.final_nodes:
                result += '{} is final\n'.format(curr_state)
                continue

            # this may need improved, the cost of this may be too high
            for state, action in self.moves.keys():
                if curr_state == state:
                    next_states = self.moves[state, action]
                    # add not visited state to the end of the queue
                    for s in next_states:
                        result += '{} -{}-> {}; '.format(state, action, s)
                        if s not in visited_nodes and s not in nodes_queue:
                            nodes_queue.append(s)
            result += '\n'

        return result

    @property
    def sorted_final_nodes(self):
        return sorted(self.final_nodes)

    def bond_re_to_final_state(self, state, regular_expr):
        if state not in self.final_nodes:
            raise ValueError("State is not a final state")

        if self.final_states_related_re.get(state) is not None:
            raise ValueError("State's regular expression has already been assigned")

        self.final_states_related_re[state] = regular_expr

    def add_edge(self, source, action, target):
        if self.moves.get((source, action)) is None:
            self.moves[source, action] = set()

        if action != _epsilon:
            self.non_epsilon_actions.add(action)

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

    def new_question_mark_nfa(self, nodes):
        self.add_epsilon_edge(nodes[0], nodes[1])
        return nodes

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

        return frozenset(closure_set)

    def _allocate_nodes(self):
        head_node = self._state_count
        last_node = self._state_count + 1
        self._state_count += 2
        self.nodes_count += 2

        return head_node, last_node

    def states_id_all_add(self, number):
        result = copy.deepcopy(self)
        # update state count
        result._state_count += number

        # update start_node
        result.start_node += number

        # update final nodes
        final_nodes = set()
        for node in result.final_nodes:
            final_nodes.add(node + number)
        result.final_nodes = final_nodes

        # update moves
        new_moves = {}
        for (from_state, action), to_state in result.moves.items():
            new_to_state = set()
            for s in to_state:
                new_to_state.add(s + number)
            new_moves[from_state + number, action] = new_to_state
        result.moves = new_moves

        # update final nodes related regular expression
        new_relations = {}
        for node, regular_expr in result.final_states_related_re.items():
            new_relations[node + number] = regular_expr
        result.final_states_related_re = new_relations
        return result

    def merge(self, others):
        result = copy.deepcopy(self)
        for other in others:
            assert isinstance(other, NFA)
            other = other.states_id_all_add(result._state_count)
            result._state_count += (other._state_count + 1)

            # add an epsilon link between result.start_node and other.start_node
            result.add_epsilon_edge(result.start_node, other.start_node)

            result.final_nodes.update(other.final_nodes)
            result.moves.update(other.moves)
            result.nodes_count += other.nodes_count
            result.final_states_related_re.update(other.final_states_related_re)
            result.non_epsilon_actions.update(other.non_epsilon_actions)

        return result


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

    for i in range(len(postfix)):
        ch = postfix[i]
        if i >= 1 and postfix[i - 1] == '\\':
            stack.append(nfa.new_single_action(ch))

        if ch == '\\':
            continue

        if is_action(ch):
            # ch is an action
            stack.append(nfa.new_single_action(ch))
        elif ch == '?':
            assert len(stack) > 0
            head, tail = stack.pop()
            stack.append(nfa.new_question_mark_nfa((head, tail)))
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
    nfa.final_nodes.add(tail)
    nfa.bond_re_to_final_state(tail, regular_expr)

    return nfa


def single_action_nfa(action):
    nfa = NFA()
    head, tail = nfa.new_single_action(action)
    nfa.start_node = head
    nfa.final_nodes.add(tail)
    nfa.bond_re_to_final_state(tail, action)

    return nfa


def integrate_multi_nfa(multi_nfa):
    result = NFA()
    result.start_node = 'x'
    return result.merge(multi_nfa)


# ------------------------------------------------------------
# construct dfa


class DFA:

    def __init__(self):
        self.start_state = None
        self.non_final_states = set()
        self.final_states = set()
        self.moves = {}
        self.final_states_related_re = {}
        self.actions = set()

        self._states_id = {}
        self._state_count = 0

    def __getitem__(self, *state_action):
        return self.moves.get(state_action, None)

    def _bond_re_to_final_states(self, state, regular_expr):
        if state not in self.final_states:
            raise ValueError("State is not a final state")

        if self.final_states_related_re.get(state) is not None:
            return

        self.final_states_related_re[state] = regular_expr

    def _get_id_for_states(self, states):
        # allocate an id for the states if it's not recorded yet
        # otherwise just return the recorded id
        states = frozenset(states)
        allocated_id = self._states_id.get(states)

        if allocated_id is None:
            allocated_id = self._state_count
            self._state_count += 1
            self._states_id[states] = allocated_id
        return allocated_id

    @property
    def nodes_count(self):
        return len(self.non_final_states) + len(self.final_states)

    def parse_string(self, string):
        tokens = []
        curr_state = self.start_state
        pointer = 0
        while pointer < len(string):
            ch = string[pointer]
            next_state = self.__getitem__(curr_state, ch)
            if next_state is None:
                token = self.get_state_related_re(curr_state)
                tokens.append(self.get_state_related_re(curr_state))

                if token == INVALID_CHARACTER:
                    return tokens

                curr_state = self.start_state
                pointer -= 1
            else:
                curr_state = next_state

            pointer += 1

        tokens.append(self.get_state_related_re(curr_state))

        return tokens

    def get_state_related_re(self, curr_state):
        if curr_state in self.final_states:
            return self.final_states_related_re[curr_state]
        else:
            return INVALID_CHARACTER

    def add_edge(self, source, action, target):
        if self.moves.get((source, action)) is None:
            self.moves[source, action] = set()

        self.moves[source, action].add(target)
        self.actions.add(action)

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

        start_states = nfa.epsilon_closure((nfa.start_node,))  # epsilon_closure returns a tuple of states
        state_queue = deque()  # this queue should only contains tuple of state
        state_queue.append(start_states)
        visited_states = set()

        dfa = cls()
        dfa.start_state = dfa._get_id_for_states(start_states)
        dfa.non_final_states.add(dfa.start_state)

        # update dfa actions
        dfa.actions = nfa.non_epsilon_actions

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
                for existed_states in dfa._states_id.keys():
                    if states.issubset(existed_states):
                        states = existed_states
                        break
                else:
                    states = nfa.epsilon_closure(states)

                states_id = dfa._get_id_for_states(states)
                dfa.moves[curr_state_id, action] = states_id

                # see if it's a final state, if this state contains more
                # than one final state of NFA, then choose the regular
                # expression of the final state with minimum id in the
                # nfa.final_nodes as this DFA final state's related
                # regular expression
                for final_state in nfa.sorted_final_nodes:
                    if final_state in states:
                        dfa.final_states.add(states_id)
                        dfa._bond_re_to_final_states(states_id, nfa.final_states_related_re[final_state])
                        break
                else:
                    dfa.non_final_states.add(states_id)

                # add not visited states to state_queue
                if states not in visited_states:
                    state_queue.append(states)

        return dfa._simplify()

    def _simplify(self):
        # initially divide final states into groups
        final_states_groups = {}
        for state, regular_expr in self.final_states_related_re.items():
            if final_states_groups.get(regular_expr) is None:
                final_states_groups[regular_expr] = []

            final_states_groups[regular_expr].append(state)

        # add non-final states to groups
        groups = [self.non_final_states]

        for regular_expr, states in final_states_groups.items():
            groups.append(set(states))

        prev_groups = []
        while prev_groups != groups:
            prev_groups = groups
            new_groups = []
            for i in range(len(groups)):
                group = groups[i]
                # just divide a group to two small groups in every iteration
                group_1 = set()
                group_2 = set()
                flag_node = next(iter(group), None)  # fetch first node of this group as the flag
                if flag_node is None:
                    print("Oh shit")
                # test action on nodes of one group
                test_groups = _construct_test_group(new_groups, groups, i)

                for node in group:
                    if all(_weak_equivalent_on_action(self.__getitem__(node, action),
                                                      self.__getitem__(flag_node, action),
                                                      test_groups)
                           for action in self.actions):
                        group_1.add(node)
                    else:
                        group_2.add(node)

                new_groups.append(group_1)
                if len(group_2) > 0:
                    new_groups.append(group_2)

            groups = new_groups

        # construct simplified dfa
        simplified_dfa = DFA()
        for group in groups:
            states_id = simplified_dfa._get_id_for_states(group)

            if self.start_state in group:
                simplified_dfa.start_state = states_id
                simplified_dfa.non_final_states.add(states_id)
            elif group.issubset(self.final_states):
                simplified_dfa.final_states.add(states_id)
                # add related regular expression to simplified dfa
                node = next(iter(group))
                assert node in self.final_states
                simplified_dfa.final_states_related_re[states_id] = self.final_states_related_re[node]
            else:
                simplified_dfa.non_final_states.add(states_id)

        simplified_dfa.actions = set(self.actions)

        for action in self.actions:
            for group in groups:
                # fetching one node is enough
                node = next(iter(group))
                assert node is not None
                # establish new links between new states
                target_state = self.__getitem__(node, action)
                if target_state is not None:
                    # find target_state's group
                    target_id = simplified_dfa._get_id_for_states(_get_group(target_state, groups))
                    simplified_dfa.moves[simplified_dfa._get_id_for_states(group), action] = target_id

        return simplified_dfa


def _get_group(state, groups):
    for group in groups:
        if state in group:
            return group
    assert False


def _weak_equivalent_on_action(state, flag_state, groups):
    if state == flag_state:
        return True
    elif state is None or flag_state is None:
        return False
    else:
        flag_group = next((group for group in groups if flag_state in group), None)
        assert flag_group is not None
        return state in flag_group


def _construct_test_group(new_groups, old_groups, old_groups_split_index):
    return new_groups + old_groups[old_groups_split_index:]


# -------------------------------------------------------------
# process input and output

class Lex:

    def __init__(self, regular_expressions, regex_token_dict):
        self.regular_expressions = regular_expressions
        self.regex_token_dict = regex_token_dict
        self._dfa = None

        self._fit()

    def _fit(self):
        multi_nfa = []
        for regular_expr in self.regular_expressions:
            if len(regular_expr) == 1:
                multi_nfa.append(single_action_nfa(regular_expr))
            else:
                multi_nfa.append(construct_nfa(regular_expr))
        nfa = integrate_multi_nfa(multi_nfa)
        self._dfa = DFA.subset_construction(nfa)

    def parse(self, characters):

        parsed_result = self._dfa.parse_string(characters)
        result_tokens = []
        for regular_expr in parsed_result:
            if regular_expr == INVALID_CHARACTER:
                raise ValueError("Invalid input")
            else:
                result_tokens.append(self.regex_token_dict[regular_expr])

        return result_tokens
