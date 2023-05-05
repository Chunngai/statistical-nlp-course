from typing import List


class Parser:
    """A toy parser with basic nlp functionalities."""

    def __init__(self):
        ...

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """Text tokenization.

        :param text: (str) text to tokenize.
        :return: (List[str]) tokens of the text.
        """

        # TODO: Replace with the implemented method.
        return ["包含", "基础", "自然语言处理", "功能", "的", "简单", "分词器", "。"]

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
    tokens = parser.tokenize(text=text)
    print(tokens)
    # Pos tagging.
    pos_list = parser.pos_tag(tokens=tokens)
    print(pos_list)
    # Syntax parsing.
    constituency_root = parser.constituency_parse(text=text)
    print(constituency_root)
