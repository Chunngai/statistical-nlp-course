#!/usr/bin/env python
# coding: utf-8


import sys
import os
import argparse
from typing import List, Tuple

from one_gram_tokenizer import OneGramTokenizer

sys.path.append("..")
join = os.path.join


class HMMPosTagger:
    def __init__(self, observable_states: List[str] = None, hidden_states: List[str] = None,
                 transition_probabilities: List[List[float]] = None, emission_probabilities: List[List[float]] = None,
                 start_probabilities: List[float] = None,
                 hmm_model_dir: str = ""):
        """Constructor of the tagger.

        :param observable_states: Observable states of the hmm model.
        :param hidden_states: Hidden states of the hmm model.
        :param transition_probabilities: Transition probs of the hmm model.
        :param emission_probabilities: Emission probs of the hmm model.
        :param start_probabilities: Start probs of the hmm model.
        :param hmm_model_dir: Dir path of the hmm model.

        If one of the components of the hmm model is provided,
        other components should also be provided. Or the dir path
        of the hmm model should be provided.
        """

        if observable_states is not None:
            assert all([hidden_states,
                        transition_probabilities, emission_probabilities, start_probabilities]) and hmm_model_dir == ""

            self.observable_states: List[str] = observable_states
            self.hidden_states: List[str] = hidden_states
            self.transition_probabilities: List[List[float]] = transition_probabilities
            self.emission_probabilities: List[List[float]] = emission_probabilities
            self.start_probabilities: List[float] = start_probabilities
        else:
            self.hmm_model_dir = hmm_model_dir
            self.hmm_model_name = os.path.split(hmm_model_dir)[-1]

            self.observable_states, self.hidden_states, \
            self.transition_probabilities, self.emission_probabilities, \
            self.start_probabilities = self._get_components()

    def _get_components(self):
        """Get components of the hmm model."""

        def _str2float(prob_list):
            """Convert a str list to a float list.

            :param prob_list: A list containing strings.
            :return: A list containing floats.
            """

            return [float(prob) for prob in prob_list]

        file_name = join(self.hmm_model_dir, self.hmm_model_name)

        with open(f"{file_name}.observable.states", encoding="utf-8") as f:
            observable_states = f.readline().split()

        with open(f"{file_name}.hidden.states", encoding="utf-8") as f:
            hidden_states = f.readline().split()

        transition_probs = []
        with open(f"{file_name}.transition.probs", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                prob_list = _str2float(line.split())
                transition_probs.append(prob_list)

        emission_probs = []
        with open(f"{file_name}.emission.probs", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                prob_list = _str2float(line.split())
                emission_probs.append(prob_list)

        with open(f"{file_name}.start.probs", encoding="utf-8") as f:
            start_probs = _str2float(f.readline().split())

        return observable_states, hidden_states, transition_probs, emission_probs, start_probs

    @staticmethod
    def _display_dp_matrix(dp_matrix):
        """Visualize the DP matrix."""

        for time_step in range(len(dp_matrix)):  # Time steps.
            for hidden_state_index in range(len(dp_matrix[time_step])):  # Hidden states.
                print(f"{dp_matrix[time_step][hidden_state_index]:.10f}", end=" ")
            print()

    def tag(self, tokens: List[str], verbose: bool = False) -> List[Tuple[str, str]]:
        """Tagging using hmm.

        :param tokens: Tokens to be tagged.
        :param verbose: Displays the dp matrix if set to True.
        :return: A dict in the form of {word: pos}.
        """

        tokens_number = len(tokens)
        hidden_states_number = len(self.hidden_states)

        oov_prob = 1 / hidden_states_number

        # Inits the dp matrix.
        dp_matrix = [
            [0.0 for _ in range(hidden_states_number)]
            for _ in range(tokens_number)
        ]

        # Stores the tagging result.
        result = []

        # Viterbi algorithm.
        for time_step in range(tokens_number):
            current_token = tokens[time_step]

            for hidden_state_index in range(hidden_states_number):
                # Handles OOV problem.
                try:
                    observable_state_index = self.observable_states.index(current_token)
                except ValueError:
                    dp_matrix[time_step][hidden_state_index] = oov_prob
                    continue

                if time_step == 0:
                    dp_matrix[time_step][hidden_state_index] = self.start_probabilities[hidden_state_index] * \
                                                               self.emission_probabilities[hidden_state_index][
                                                                   observable_state_index]
                else:
                    dp_matrix[time_step][hidden_state_index] = max([
                        dp_matrix[time_step - 1][previous_hidden_state_index] *
                        self.transition_probabilities[previous_hidden_state_index][hidden_state_index] *
                        self.emission_probabilities[hidden_state_index][observable_state_index]

                        for previous_hidden_state_index in range(hidden_states_number)
                    ])

            # Gets the most likely pos at the current time step.
            max_hidden_state_probability = max(dp_matrix[time_step])
            index_of_max_hidden_state_probability = dp_matrix[time_step].index(max_hidden_state_probability)
            hidden_state_of_max_probability = self.hidden_states[index_of_max_hidden_state_probability]

            result.append(
                (current_token, hidden_state_of_max_probability)
            )

        if verbose:
            self._display_dp_matrix(dp_matrix)

        return result


if __name__ == '__main__':
    pass
