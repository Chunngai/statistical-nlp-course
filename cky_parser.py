from typing import List


class Node:
    def __init__(self, parent, lchild, rchild, prob):
        """Construct a node of the parsing tree.

        :param parent: Non-terminals.
        :param lchild: Non-terminals or terminals.
        :param rchild: Non-terminals or None.
        :param prob: Prob of the rule: parent -> lchild (rchild).

        E.g.,
        S -> NP VP  0.9: parent=S, lchild=NP, rchild=VP, prob=0.9
        S -> VP     0.1: parent=S, lchild=VP, rchild=None, prob=0.1
        N -> people 0.5: parent=N, lchild=people, rchild=None, prob=0.5

        """

        self.parent = parent
        self.lchild = lchild
        self.rchild = rchild

        self.prob = prob

    def __str__(self):
        """Formatted representation: parent->lchild (rchild) prob"""

        if type(self.lchild) == str:
            lchild = self.lchild
        else:
            lchild = self.lchild.parent

        if self.rchild is not None:
            if type(self.rchild) == str:
                rchild = self.rchild
            else:
                rchild = self.rchild.parent
        else:
            rchild = ""

        return f"{self.parent}->{lchild} {rchild} {self.prob:.2e}"


class CKYParser:
    def __init__(self, T, N, S, R, P):
        """Constructor of a cky parser.

        :param T: Terminals.
        :param N: Non-terminals.
        :param S: Start-of-sentence token in non-terminals.
        :param R: Rules.
        :param P: Probs.
        """

        # Terminals in T should be unique.
        assert len(set(T)) == len(T)
        # Non-terminals in N should be unique.
        assert len(set(N)) == len(N)
        # Lengths of R and P should match.
        assert len(R) == len(P)

        self.T = T
        self.N = N
        self.S = S
        self.R = R
        self.P = P

        self.nodes = self._make_nodes()
        # List of rules in the form of non-terminal -> non-terminal.
        self.unaries = self._get_unaries()

    def _make_nodes(self):
        """Convert each rule into a node of the parsing tree.

        :return: List of nodes.
        """

        nodes = []
        for prob, rule in zip(self.P, self.R):
            rule_split = rule.split("->")  # E.g., ["S", "NP VP"]

            parent = rule_split[0]  # "S".

            children = rule_split[1].split()  # ["NP", "VP"].
            if len(children) == 2:
                lchild = children[0]
                rchild = children[1]
            else:
                lchild = children[0]
                rchild = None

            node = Node(
                parent=parent,
                lchild=lchild,
                rchild=rchild,
                prob=prob
            )
            nodes.append(node)

        return nodes

    def _get_unaries(self):
        """Get all rules in the form of non-terminal->non-terminal.

        :return: List of such rules.
        """

        unaries = []
        for node in self.nodes:
            if node.lchild in self.N and node.rchild is None:  # E.g., S -> VP.
                unaries.append(node)

        return unaries

    def parse(self, tokens: List[str]):
        """Parse a sentence.

        :param tokens: The sentence to parse.
        :return: Root of the parsing tree.
        """

        def _visualize_dp_matrix():
            """Visualize the dp matrix."""

            for i in range(n_tokens):
                for j in range(n_tokens):
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

            print(f"{'  ' * count} ({node.parent}, {node.prob:.2e})")

            count += 1
            _visualize_parsing_tree(node.lchild, count)
            _visualize_parsing_tree(node.rchild, count)

        def _handle_unaries(node, row_idx, col_idx):
            """Handle unaries.

            :param node: non-terminal2 in non-terminal1 -> non-terminal2.
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
                    if current_node.parent == unary.lchild:
                        prob = unary.prob * current_node.prob

                        # E.g.,
                        # Node1: S -> NP VP 1.26e-3.
                        # Node2: S -> VP 1.05e-2.
                        # Then keep node2 and discard node1.
                        try:
                            if prob > current_cell[unary.parent].prob:
                                current_cell[unary.parent] = Node(
                                    parent=unary.parent,
                                    lchild=current_node,
                                    rchild=None,
                                    prob=prob
                                )
                                added = True
                                current_node = current_cell[unary.parent]
                        except KeyError:
                            current_cell[unary.parent] = Node(
                                parent=unary.parent,
                                lchild=current_node,
                                rchild=None,
                                prob=prob
                            )
                            added = True
                            current_node = current_cell[unary.parent]

        n_tokens = len(tokens)

        # 3d matrix. dp_matrix[i][j] stores all possible nodes.
        dp_matrix = [
            [
                {}  # dp_matrix[i][j].
                for _ in range(n_tokens)
            ] for _ in range(n_tokens)
        ]

        # i == j.
        for i in range(n_tokens):
            token = tokens[i]
            for node in self.nodes:
                if node.lchild == token:  # E.g., N -> people.
                    # \pi(i, i, X) = q(X -> w_i)
                    dp_matrix[i][i][node.parent] = node

                    # Iteratively handles unaries.
                    _handle_unaries(
                        node=node,
                        row_idx=i,
                        col_idx=i
                    )

        # i < j
        for span in range(1, n_tokens):
            for begin in range(n_tokens - span):
                end = begin + span

                """
                begin end
                
                0     1
                1     2
                2     3
                
                0     2
                1     3
                
                0     3
                """

                for split in range(begin, end):
                    # E.g.,
                    # Split 1: (fish people) fish
                    # Split 2: fish (people fish)
                    for node in self.nodes:
                        if node not in self.unaries and \
                                node.lchild in dp_matrix[begin][split].keys() and \
                                node.rchild in dp_matrix[split + 1][end].keys():

                            # \pi(i, j, X) = max(q(X -> YZ) * \pi(i, k, Y) * \pi(k+1, j, Z)), i<=k<=j-1
                            prob = dp_matrix[begin][split][node.lchild].prob * \
                                   dp_matrix[split + 1][end][node.rchild].prob * \
                                   node.prob
                            try:
                                if prob > dp_matrix[begin][end][node.parent].prob:
                                    dp_matrix[begin][end][node.parent] = Node(
                                        parent=node.parent,
                                        lchild=dp_matrix[begin][split][node.lchild],
                                        rchild=dp_matrix[split + 1][end][node.rchild],
                                        prob=prob,
                                    )
                            except KeyError:
                                dp_matrix[begin][end][node.parent] = Node(
                                    parent=node.parent,
                                    lchild=dp_matrix[begin][split][node.lchild],
                                    rchild=dp_matrix[split + 1][end][node.rchild],
                                    prob=prob,
                                )

                            # Iteratively handles unaries.
                            _handle_unaries(dp_matrix[begin][end][node.parent], begin, end)

        # Visualizes the dp matrix.
        _visualize_dp_matrix()

        # Visualizes the parsing tree.
        root = dp_matrix[0][n_tokens - 1]["S"]
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
