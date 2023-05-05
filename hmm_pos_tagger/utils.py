import copy
import os
from typing import List

join = os.path.join


def preprocess_199801(file_path: str):
    """Generate a hmm model from the 199801.txt file.

    :param file_path: Path of "199801.txt".
    :return: None.

    Observable states, hidden states, transition probabilities, emission probabilities and start probabilities
    are generated from the file and are saved in a dir named "199801".
    """

    def _add_to_states(element, states):
        if element in padding_stuffs:
            return

        if element not in states:
            states.append(element)

    def _add_to_freqs_dict(a, b, probs_dict):
        if a in padding_stuffs or b in padding_stuffs:
            return

        key_tuple = (a, b)
        if key_tuple not in probs_dict.keys():
            probs_dict[key_tuple] = 1
        else:
            probs_dict[key_tuple] += 1

    def _add_to_start_freqs(pos, start_freqs):
        if pos in padding_stuffs:
            return

        if pos not in hidden_states:
            start_freqs.append(1)
        else:
            pos_index = hidden_states.index(pos)
            start_freqs[pos_index] += 1

    def _generate_probs(rows, cols, freqs_dict):
        row_number = len(rows)
        col_number = len(cols)

        freqs = [
            [0.0 for _ in range(col_number)]
            for _ in range(row_number)
        ]
        probs = copy.deepcopy(freqs)

        for row_index in range(row_number):
            for col_index in range(col_number):
                a = rows[row_index]
                b = cols[col_index]
                key_tuple = (a, b)

                try:
                    freqs[row_index][col_index] = freqs_dict[key_tuple]
                    # +1: smoothing.
                    probs[row_index][col_index] = freqs_dict[key_tuple] + 1
                except KeyError:
                    freqs[row_index][col_index] = 0
                    # +1: smoothing.
                    probs[row_index][col_index] = 1

            row_sum = sum(probs[row_index])
            for col_index in range(col_number):
                probs[row_index][col_index] /= row_sum

        return freqs, probs

    def _save():
        def _float2str(prob_list):
            return [f"{prob:.12f}" for prob in prob_list]

        dir_name = file_path.split('.')[0]
        file_name = dir_name

        os.makedirs(dir_name, exist_ok=True)

        with open(f"{join(dir_name, file_name)}.observable.states", 'w', encoding="utf-8") as f:
            f.write(" ".join(observable_states))

        with open(f"{join(dir_name, file_name)}.hidden.states", 'w', encoding="utf-8") as f:
            f.write(" ".join(hidden_states))

        with open(f"{join(dir_name, file_name)}.transition.probs", 'w', encoding="utf-8") as f:
            for row in transition_probs:
                row_ = _float2str(row)
                f.write(" ".join(row_))
                f.write("\n")

        with open(f"{join(dir_name, file_name)}.emission.probs", 'w', encoding="utf-8") as f:
            for row in emission_probs:
                row_ = _float2str(row)
                f.write(" ".join(row_))
                f.write("\n")

        with open(f"{join(dir_name, file_name)}.start.probs", 'w', encoding="utf-8") as f:
            row_ = _float2str(start_probs)
            f.write(" ".join(row_))

    observable_states: List[str] = []  # All tokens.
    hidden_states: List[str] = []  # All parts of speech.

    transition_freqs_dict = {}
    emission_freqs_dict = {}

    start_freqs: List[float] = []

    with open(file_path, encoding="utf-8") as file:
        lines = file.readlines()

    start_token = '<start_token>'
    start_pos = '<start_pos>'
    end_token = '<end_token>'
    end_pos = '<end_pos>'
    padding_stuffs = [start_token, start_pos, end_token, end_pos]

    for line in lines:
        line = f"{start_token}/{start_pos} {line} {end_token}/{end_pos}"

        token_pos_list = line.split()

        _, prev_pos = token_pos_list[0].split('/')
        for token_pos in token_pos_list[1:]:
            token, pos = token_pos.split("/")

            _add_to_start_freqs(pos, start_freqs)

            _add_to_states(token, observable_states)
            _add_to_states(pos, hidden_states)

            _add_to_freqs_dict(prev_pos, pos, transition_freqs_dict)
            _add_to_freqs_dict(pos, token, emission_freqs_dict)

            prev_pos = pos

    # prev_pos -> fol_pos. N(prev_pos fol_pos) / N(prev_pos *)
    transition_freqs, transition_probs = _generate_probs(hidden_states, hidden_states, transition_freqs_dict)

    # pos -> token.  N(pos, token) / N(pos)
    emission_freqs, emission_probs = _generate_probs(hidden_states, observable_states, emission_freqs_dict)

    start_freqs_sum = sum(start_freqs)
    start_probs = []
    for pos_index in range(len(hidden_states)):
        start_probs.append(start_freqs[pos_index] / start_freqs_sum)

    _save()


if __name__ == '__main__':
    preprocess_199801("199801.txt")