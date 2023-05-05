from typing import List


class Node:
    def __init__(self, prob, value, lchild, rchild):
        """Construct a node of the parsing tree.

        :param prob: Prob of the rule: value->lchild (rchild).
        :param value: Non-terminals.
        :param lchild: Non-terminals or terminals.
        :param rchild: Non-terminals or None.
        """

        self.prob = prob
        self.value = value
        self.lchild = lchild
        self.rchild = rchild

    def __str__(self):
        """Formatted representation: value->lchild (rchild) prob"""

        if type(self.lchild) == str:
            lchild = self.lchild
        else:
            lchild = self.lchild.value

        if self.rchild is not None:
            if type(self.rchild) == str:
                rchild = self.rchild
            else:
                rchild = self.rchild.value
        else:
            rchild = ""

        return f"{self.value}->{lchild} {rchild} {self.prob:.2e}"


class CKYParser:
    def __init__(self, T, N, S, R, P):
        """Constructor of a cky parser.

        :param T: Terminals.
        :param N: Non-terminals.
        :param S: Start-of-sentence token in non-terminals.
        :param R: Rules.
        :param P: Probs.
        """

        # Terminals in the list should be unique.
        assert len(set(T)) == len(T)
        # Non-terminals in the set should be unique.
        assert len(set(N)) == len(N)
        assert len(R) == len(P)

        self.T = T
        self.N = N
        self.S = S
        self.R = R
        self.P = P

        # Each node is like: value->lchild rchild   prob.
        self.nodes = self._make_nodes()
        # List of rules in the form of non-terminal->non-terminal.
        self.unaries = self._get_unaries()

    def _make_nodes(self):
        """Convert each rule into a node of the parsing tree.

        :return: List of nodes.
        """

        nodes = []
        for prob, rule in zip(self.P, self.R):
            rule_split = rule.split("->")
            from_ = rule_split[0]
            to_ = rule_split[1].split()

            node = Node(prob=prob, value=from_, lchild=to_[0], rchild=to_[1] if len(to_) == 2 else None)
            nodes.append(node)

        return nodes

    def _get_unaries(self):
        """Get all rules in the form of non-terminal->non-terminal.

        :return: List of such rules.
        """

        unaries = []
        for node in self.nodes:
            if node.lchild in self.N and node.rchild is None:
                unaries.append(node)

        return unaries

    def parse(self, sentence: List[str]):
        """Parse a sentence.

        :param sentence: The sentence to parse.
        :return: Root of the parsing tree.
        """

        def _visualize_dp_matrix():
            """Visualize the dp matrix."""

            for i in range(sentence_len):
                for j in range(sentence_len):
                    print("(", end="")
                    for k in dp_matrix[i][j].keys():
                        node = dp_matrix[i][j][k]
                        print(node, end=", ")
                    print("), ", end="")
                print()

        def _visualize_parsing_tree(node, count):
            """Visualize the parsing tree.

            :param node: A node in the parsing tree.
            :param count: Number of indentation.
            :return: None.
            """

            # Boundaries of recursion.
            if node is None:  # rchild.
                return
            if type(node) == str:  # lchild, terminals.
                print(f"{'  ' * (count + 1)} {node}")
                return

            print(f"{'  ' * count} ({node.value}, {node.prob:.2e})")

            count += 1
            _visualize_parsing_tree(node.lchild, count)
            _visualize_parsing_tree(node.rchild, count)

        def _handle_unaries(node, row_idx, col_idx):
            """Handle unaries.

            :param node: non-terminal2 in "non-terminal1->non-terminal2".
            :param row_idx: row index of the current cell in the dp matrix.
            :param col_idx: col index of the current cell in the dp matrix.
            :return: None.
            """

            current_cell = dp_matrix[row_idx][col_idx]
            current_node = node
            added = True
            while added:
                added = False

                for unary in self.unaries:
                    # Finds a node in the cell whose rule is a unary.
                    if current_node.value == unary.lchild:
                        prob = unary.prob * current_node.prob
                        try:
                            if prob > current_cell[unary.value].prob:
                                current_cell[unary.value] = Node(prob=prob, value=unary.value,
                                                                 lchild=current_node, rchild=None)
                                added = True
                                current_node = current_cell[unary.value]
                        except KeyError:
                            current_cell[unary.value] = Node(prob=prob, value=unary.value,
                                                             lchild=current_node, rchild=None)
                            added = True
                            current_node = current_cell[unary.value]

        sentence_len = len(sentence)

        # 3d matrix. dp_matrix[i][j] stores all possible nodes.
        dp_matrix = [[{} for _ in range(sentence_len)] for _ in range(sentence_len)]

        # i == j.
        for i in range(sentence_len):
            word = sentence[i]
            for node in self.nodes:
                if node.lchild == word:
                    # \pi(i, i, X) = q(X -> w_i)
                    dp_matrix[i][i][node.value] = node

                    # Iteratively handles unaries.
                    _handle_unaries(node, i, i)

        # i < j
        for span in range(1, sentence_len):
            for begin in range(sentence_len - span):
                end = begin + span

                for split in range(begin, end):
                    for node in self.nodes:
                        if node not in self.unaries and \
                                node.lchild in dp_matrix[begin][split].keys() and \
                                node.rchild in dp_matrix[split + 1][end].keys():

                            # \pi(i, j, X) = max(q(X -> YZ) * \pi(i, k, Y) * \pi(k+1, j, Z)), i<=k<=j-1
                            prob = dp_matrix[begin][split][node.lchild].prob * \
                                   dp_matrix[split + 1][end][node.rchild].prob * \
                                   node.prob
                            try:
                                if prob > dp_matrix[begin][end][node.value].prob:
                                    dp_matrix[begin][end][node.value] = Node(
                                        prob=prob,
                                        value=node.value,
                                        lchild=dp_matrix[begin][split][node.lchild],
                                        rchild=dp_matrix[split + 1][end][node.rchild]
                                    )
                            except KeyError:
                                dp_matrix[begin][end][node.value] = Node(
                                        prob=prob,
                                        value=node.value,
                                        lchild=dp_matrix[begin][split][node.lchild],
                                        rchild=dp_matrix[split + 1][end][node.rchild]
                                    )

                            # Iteratively handles unaries.
                            _handle_unaries(dp_matrix[begin][end][node.value], begin, end)

        # Visualizes the dp matrix.
        _visualize_dp_matrix()

        # Visualizes the parsing tree.
        root = dp_matrix[0][sentence_len - 1]["S"]
        _visualize_parsing_tree(root, 0)

        return root


if __name__ == '__main__':
    cky_parser = CKYParser(
        T=["people", "fish", "tanks", "rods", "with"],
        N=["S", "NP", "VP", "V", "@VP_V", "PP", "N", "P"],
        S="S",
        R=['S->NP VP', 'S->VP', 'VP->V NP', 'VP->V', 'VP->V @VP_V', 'VP->V PP', '@VP_V->NP PP', 'NP->NP NP',
           'NP->NP PP', 'NP->N', 'PP->P NP', 'N->people', 'N->fish', 'N->tanks', 'N->rods', 'V->people', 'V->fish',
           'V->tanks', 'P->with'],
        P=[0.9, 0.1, 0.5, 0.1, 0.3, 0.1, 1.0, 0.1, 0.2, 0.7, 1.0, 0.5, 0.2, 0.2, 0.1, 0.1, 0.6, 0.3, 1.0]
    )

    root = cky_parser.parse(["fish", "people", "fish", "tanks"])
