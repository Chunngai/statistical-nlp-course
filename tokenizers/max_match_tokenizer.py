import os.path
from typing import List


def read_vocabulary():
    """Read the vocabulary."""

    fp_vocab = "data/vocab.txt"
    if os.path.exists(fp_vocab):
        with open(fp_vocab, encoding="utf-8") as f:
            vocab = eval(f.read())
    else:
        # Construct the vocabulary.
        vocab = set()
        with open("data/199801.txt", "r", encoding="utf-8") as f:
            for line in f.read().strip().splitlines():  # Read each line.
                for word in line.split(" "):
                    vocab.add(word.split("/")[0])  # Read each word.

        # Save the vocabulary.
        with open(fp_vocab, "w", encoding="utf-8") as f:
            f.write(str(vocab))

    return vocab


vocabulary = read_vocabulary()


def mm_tokenize(text: str, max_len: int = 3):
    """Max match tokenization.

    :param text: (str) text to tokenize.
    :param max_len: (int) max word length.
    :return: (List[str]) tokenized words.
    """

    text = "".join(text.split())  # Remove spaces.
    word_list = []
    i = 0  # Current index (count from the start).
    while i < len(text):
        longest_word = text[i]
        for j in range(i + 1, i + max_len + 1):
            word = text[i:j]
            if word in vocabulary:
                if len(word) > len(longest_word):
                    longest_word = word
        word_list.append(longest_word)
        i += len(longest_word)
    return word_list


def rmm_tokenize(text: str, max_len: int = 3):
    """Reversed max match tokenization.

    :param text: (str) text to tokenize.
    :param max_len: (int) max word length.
    :return: (List[str]) words.
    """

    text = "".join(text.split())  # Remove spaces.
    word_list = []
    i = len(text) - 1  # Current index (count from the end).
    while i >= 0:
        longest_word = text[i]
        for j in range(i - max_len + 1, i):
            word = text[j:i + 1]
            if word in vocabulary:
                if len(word) > len(longest_word):
                    longest_word = word
        word_list.insert(0, longest_word)
        i -= len(longest_word)
    return word_list


def bmm_tokenize(text: str, max_len: int = 3):
    """Bidirectional max match tokenization.

    :param text: (str) text to tokenize.
    :param max_len: (int) max word length.
    :return: (List[str]) words.
    """

    def count_single_words(words: List[str]):
        """Count the number of single words.

        :param words: (List[str]) Tokenized words.
        :return: (int) number of single words.
        """

        result = 0
        for i in text:
            if len(i) == 1:
                result += 1
        return result

    text = "".join(text.split())  # Remove spaces.
    list_forward = mm_tokenize(text, max_len)
    list_backward = rmm_tokenize(text, max_len)

    # Select the result with less words.
    if len(list_forward) > len(list_backward):
        list_final = list_backward[:]
    elif len(list_forward) < len(list_backward):
        list_final = list_forward[:]
    else:
        # Select the result with less single words.
        if count_single_words(list_forward) > count_single_words(list_backward):
            list_final = list_backward[:]
        elif count_single_words(list_forward) < count_single_words(list_backward):
            list_final = list_forward[:]
        else:
            list_final = list_backward[:]
    return list_final


if __name__ == "__main__":
    text = "这是一个管弦乐乐团。"

    mm_tokens = mm_tokenize(text)
    rmm_tokens = rmm_tokenize(text)
    bmm_tokens = bmm_tokenize(text)
    print("正向最长匹配", mm_tokens)
    print("逆向最长匹配", rmm_tokens)
    print("双向最长匹配", bmm_tokens)
