from typing import List

from pos_taggers.hmm_pos_tagger import HmmPosTagger
from tokenizers.rule_based_tokenizer import mm_tokenize, rmm_tokenize, bmm_tokenize


class StatisticalChineseParser:
    """A statistical Chinese parser with basic nlp functionalities."""

    def __init__(self):
        self.tokenizers = {
            "mm": mm_tokenize,
            "rmm": rmm_tokenize,
            "bmm": bmm_tokenize,
        }

        self.pos_tagger = HmmPosTagger()

    def tokenize(self, text: str, tokenizer: str = "bmm") -> List[str]:
        """Text tokenization.

        :param text: (str) text to tokenize.
        :param tokenizer: (str) name of the tokenizer.
        :return: (List[str]) tokens of the text.
        """

        tokenizer = self.tokenizers[tokenizer]
        return tokenizer(text)

    def pos_tag(self, tokens: List[str]) -> List[str]:
        """Part-of-speech tagging.

        :param tokens: (List[str]) Tokens for pos tagging.
        :return: (List[str]) POS labels for the tokens.
        """

        return self.pos_tagger.predict(tokens)

    @classmethod
    def constituency_parse(cls, text: str):
        """Constituency syntax parsing.

        :param text: (str): text to parse.
        :return: (Node): root node of the parsed tree.
        """

        # TODO: Replace with the implemented method.
        return None


if __name__ == '__main__':
    text = "一个基于统计的中文文本解析器。"

    parser = StatisticalChineseParser()

    # Tokenization.
    tokens = parser.tokenize(text=text, tokenizer="bmm")
    print(tokens)

    # Pos tagging.
    pos_list = parser.pos_tag(tokens=tokens)
    print(pos_list)

    # Syntax parsing.
    constituency_root = parser.constituency_parse(text=text)
    print(constituency_root)
