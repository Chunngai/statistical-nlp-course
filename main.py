from typing import List

from tokenizers.rule_based_tokenizer import mm_tokenize, rmm_tokenize, bmm_tokenize


class Parser:
    """A toy parser with basic nlp functionalities."""

    def __init__(self):
        self.tokenizers = {
            "mm": mm_tokenize,
            "rmm": rmm_tokenize,
            "bmm": bmm_tokenize,
        }

    def tokenize(self, text: str, tokenizer="bmm") -> List[str]:
        """Text tokenization.

        :param text: (str) text to tokenize.
        :return: (List[str]) tokens of the text.
        """

        tokenizer = self.tokenizers[tokenizer]
        return tokenizer(text)

    @classmethod
    def pos_tag(cls, tokens: List[str]) -> List[str]:
        """Part-of-speech tagging.

        :param tokens: (List[str]) Tokens for pos tagging.
        :return: (List[str]) POS labels for the tokens.
        """

        # TODO: Replace with the implemented method.
        return ["VV", "JJ", "NN", "NN", "DEC", "JJ", "NN", "PU"]

    @classmethod
    def constituency_parse(cls, text: str):
        """Constituency syntax parsing.

        :param text: (str): text to parse.
        :return: (Node): root node of the parsed tree.
        """

        # TODO: Replace with the implemented method.
        return None


if __name__ == '__main__':
    text = "包含基础自然语言处理功能的简单分词器。"

    parser = Parser()
    # Tokenization.
    tokens = parser.tokenize(text=text, tokenizer="bmm")
    print(tokens)
    # Pos tagging.
    pos_list = parser.pos_tag(tokens=tokens)
    print(pos_list)
    # Syntax parsing.
    constituency_root = parser.constituency_parse(text=text)
    print(constituency_root)
