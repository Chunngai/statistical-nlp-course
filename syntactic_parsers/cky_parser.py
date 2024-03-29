from typing import List, Tuple, Set, Dict


class Rule:

    def __init__(self, head: str, ltail: str, prob: float, rtail: str = ""):
        self.head = head
        self.ltail = ltail
        self.rtail = rtail
        self.prob = prob

    def __str__(self):
        return self.head + "->" + self.ltail + " " + self.rtail + " " + str(self.prob)

    def is_terminal_rule(self, terminal_set: Set) -> bool:
        """Judge the node which is the terminal rule.

        :param terminal_set: node.value set, which should be the word.
        :return: true or false.
        """

        if self.ltail in terminal_set and self.rtail == "":
            return True
        else:
            return False

    def has_rtail(self) -> bool:
        if self.rtail != "":
            return True
        else:
            return False


class Node:

    def __init__(self, value: str, lchild, prob: float, rchild=None):
        """A class for constructing the grammar rule.

        Each Node object has 4 attributes: parent, lchild, rchild, probability (named by prob).
        The children are determined by the arrow ("->") of the grammar rule.

        Example:    S   ->  NP VP [0.9]
        :param value: string, the name of this node, Non-terminal or terminal, exists in the left of arrow (example: S).
        :param lchild: Node or None, Non-terminal (Node) or terminal (string), the first element existing in the right
            of arrow (example: NP).
        :param prob: the probability of the rule (example: 0.9).
        :param rchild: Node or None, the second element (Node) existing in the right of arrow (example: VP)
            otherwise, set as the default value (None).
        """

        self.value = value
        self.lchild = lchild
        self.rchild = rchild
        self.prob = prob

        self.LEVEL = -1

    def __str__(self):
        def inorder(node):
            nonlocal s

            if node.value is not None:
                self.LEVEL += 1
                if node.has_lchild():
                    s += f"{self.LEVEL * '  '}({node.value}, {node.prob:.2e})\n"
                else:
                    s += f"{self.LEVEL * '  '}{node.value}\n"

            if node.has_lchild():
                inorder(node.lchild)
            if node.has_rchild():
                inorder(node.rchild)

            self.LEVEL -= 1

        s = ""
        inorder(self)
        return s

    def has_rchild(self) -> bool:
        if self.rchild is not None:
            return True
        else:
            return False

    def has_lchild(self) -> bool:
        if self.lchild is not None:
            return True
        else:
            return False


class CKYParser:

    def __init__(self, rule_list: List[Rule], terminal_set: Set[str], root_value: str):
        """Initialize the dynamic programming matrix of the parser"""

        # the number of dictionaries in each inner list is the number of column in the statistics table
        # the number of inner lists is the number of row in the statistics table

        # divide the grammar rules into 3 types
        rules_for_terminal = []
        unary_nt_rules = []
        binary_nt_rules = []

        # judge the nodes. divide them into 3 types: terminal_rule, unary_nt_rule and binary_nt_rules
        for rule in rule_list:
            if rule.is_terminal_rule(terminal_set=terminal_set):
                rules_for_terminal.append(rule)
            else:
                if rule.has_rtail():
                    binary_nt_rules.append(rule)
                else:
                    unary_nt_rules.append(rule)

        # store for attributes
        self.rules_for_terminal = rules_for_terminal
        self.unary_nt_rules = unary_nt_rules
        self.binary_nt_rules = binary_nt_rules
        self.rule_list = rule_list

        # define the root node
        self.root = None
        self.root_value = root_value

    def parse(self, tokens):

        def _handle_unary_nt_rules(matrix_element: Dict):

            while True:
                # define a variable to store the length of current dictionary (matrix_element) temporarily
                element_dict_len = len(matrix_element.items())

                # capturing the existing rules
                # each item in this list is a tuple
                # e.g. : ('V', node object[this object with prob attribute])
                existing_nodes = list(matrix_element.items())

                # in PCFG, the rules like "VP -> NP", "PP -> VP" must be absent in the unary non-terminal rules set
                for item in existing_nodes:
                    for rule in self.unary_nt_rules:
                        if rule.ltail == item[0]:
                            if rule.head in matrix_element.keys():
                                # calculate the probability of the situations that the keys are same
                                # e.g. : S -> VP [0.3] in matrix_element, current rule is S -> NP [0.2]
                                current_handled_rule_prob = rule.prob * item[1].prob

                                # if stored handled unary rule's prob geq than current handled unary rule's prob
                                # then ignore it
                                if matrix_element[rule.head].prob >= current_handled_rule_prob:
                                    continue

                                # otherwise, replace it with current node
                                else:
                                    matrix_element[rule.head] = Node(
                                        value=rule.head,
                                        lchild=matrix_element[rule.ltail],
                                        prob=current_handled_rule_prob
                                    )
                                    continue

                            matrix_element[rule.head] = Node(
                                value=rule.head,
                                lchild=matrix_element[rule.ltail],
                                prob=rule.prob * item[1].prob
                            )

                if len(matrix_element.items()) == element_dict_len:
                    break

        dp_matrix = [
            [
                {}
                for _ in range(len(tokens))
            ] for _ in range(len(tokens))
        ]

        # i == j
        # fill the leaves nodes of the syntax tree (diagonal of the dp_matrix)
        for i in range(len(tokens)):
            for rule in self.rules_for_terminal:
                if tokens[i] == rule.ltail:
                    current_terminal_node = Node(
                        value=rule.ltail,
                        prob=1.0,
                        lchild=None
                    )

                    dp_matrix[i][i][rule.head] = Node(
                        value=rule.head,
                        lchild=current_terminal_node,
                        prob=rule.prob
                    )

            _handle_unary_nt_rules(dp_matrix[i][i])

        # i < j
        """
        the loop above outputs 3 rounds:
        span = 1:
            begin   end
              0      1
              1      2
              2      3

        span = 2:
            begin   end
              0      2
              1      3

        span = 3:
            begin   end
              0      3
        """
        for span in range(1, len(tokens)):
            for begin in range(len(tokens) - span):
                end = begin + span

                for split in range(begin, end):
                    # when begin = 0, end = 2 :
                    # split = 0 and 1, assume now split = 0,
                    # therefore, the element of dp_matrix[0][2] derives from
                    # dp_matrix[0(begin)][0(split)] and dp_matrix[1(split+1)][2(end)];
                    # or assume split = 1, the element of dp_matrix[0][2] derives from
                    # dp_matrix[0(begin)][1(split)] and dp_matrix[2(split+1)][2(end)]
                    for rule in self.binary_nt_rules:
                        if rule.ltail in dp_matrix[begin][split].keys() and \
                                rule.rtail in dp_matrix[split + 1][end].keys():
                            # define a variable to store the prob of current candidate element of matrix[begin][end]
                            current_prob = rule.prob * dp_matrix[begin][split][rule.ltail].prob \
                                           * dp_matrix[split + 1][end][rule.rtail].prob

                            try:
                                if current_prob > dp_matrix[begin][end][rule.head].prob:
                                    dp_matrix[begin][end][rule.head] = Node(
                                        value=rule.head,
                                        lchild=dp_matrix[begin][split][rule.ltail],
                                        rchild=dp_matrix[split + 1][end][rule.rtail],
                                        prob=current_prob
                                    )

                            except KeyError:
                                dp_matrix[begin][end][rule.head] = Node(
                                    value=rule.head,
                                    lchild=dp_matrix[begin][split][rule.ltail],
                                    rchild=dp_matrix[split + 1][end][rule.rtail],
                                    prob=current_prob
                                )

                            # handle the unary_nt_rules
                            _handle_unary_nt_rules(matrix_element=dp_matrix[begin][end])

        # store the root node
        self.root = dp_matrix[0][len(tokens) - 1][self.root_value]
        # print(self.dp_matrix[0][0]["N"].lchild)
        # print(self.dp_matrix[0][len(self.input_tokens) - 1]['S'])

        # handle the unary_nt_rules
        # print(list(self.dp_matrix[0][0].items()))

        return self.root


def read_grammar(grammar: str) -> Tuple[List[Rule], Set[str]]:
    """Read the PCFG grammar and convert each rule to Node object.

    :param grammar: a string, storing the grammar rules.
    :return: rule_list, non-terminal_list, terminal_list.
    """

    rule_list, terminal_node_list = [], []

    # the first and last lines of grammar is '\n' token
    rules = grammar.split('\n')[1:-1]
    for rule in rules:
        rule = rule.strip()

        # split the arrow
        current_value, children_and_prob = rule.split('->')
        current_value = current_value.strip()

        # extract the children and probability
        # handle the rules of terminals
        if "|" in children_and_prob:
            terminal_prob_pairs = children_and_prob.split('|')

            for pair in terminal_prob_pairs:
                pair = pair.strip()

                terminal, prob = pair.split('[')
                prob = float(prob.strip()[:-1])
                current_lchild = terminal.strip().replace("\"", "")

                current_rule = Rule(
                    head=current_value,
                    ltail=current_lchild,
                    prob=prob
                )

                rule_list.append(current_rule)
                terminal_node_list.append(current_lchild)

        # handle the rules of non-terminals
        else:
            current_children, prob = children_and_prob.split('[')
            current_children = current_children.strip()
            # remove the ']' of the probability string and convert it to float
            prob = float(prob.strip()[:-1])
            current_rchild = ""

            # a special situation, here handle the rules of terminals
            if '\"' in current_children:
                current_children = current_children.replace("\"", "")
                current_lchild = current_children
                terminal_node_list.append(current_lchild)

            # judge if the rchild whether exists or not
            elif " " in current_children:
                current_lchild, current_rchild = current_children.split(' ')
            else:
                current_lchild = current_children

            current_rule = Rule(
                head=current_value,
                ltail=current_lchild,
                rtail=current_rchild,
                prob=prob
            )
            rule_list.append(current_rule)

    return rule_list, set(terminal_node_list)


if __name__ == '__main__':
    # PCFG grammar
    grammar = """
        S   ->  NP VP [0.9]
        S   ->  VP [0.1]
        VP  ->  V NP [0.5]
        VP  ->  V [0.1]
        VP  ->  V @VP_V [0.3]
        VP  ->  V PP [0.1]
        @VP_V -> NP PP [1.0]
        NP  ->  NP NP [0.1]
        NP  ->  NP PP [0.2]
        NP  ->  N [0.7]
        PP  ->  P NP [1.0]
        V   ->  "people" [0.1] | "fish" [0.6] | "tanks" [0.3]
        N   ->  "people" [0.5] | "fish" [0.2] | "tanks" [0.2] | "rods" [0.1]
        P   ->  "with" [1.0]
    """

    # read and process the PCFG grammar
    rule_list, terminal_node_set = read_grammar(grammar=grammar)

    # init the CKYParser
    cky_parser = CKYParser(
        rule_list=rule_list,
        terminal_set=terminal_node_set,
        root_value="S"
    )

    # constituency parse
    root = cky_parser.parse(tokens=["fish", "people", "fish", "tanks"])
    print(root)
