from typing import List

from pos_taggers.hmm_pos_tagger import HmmPosTagger
from syntactic_parsers.cky_parser import CKYParser, read_grammar
from tokenizers.hmm_tokenizar import HmmTokenizer
from tokenizers.max_match_tokenizer import mm_tokenize, rmm_tokenize, bmm_tokenize


class StatisticalChineseParser:
    """A statistical Chinese parser with basic nlp functionalities."""

    def __init__(self):
        self.tokenize_funcs = {
            "mm": mm_tokenize,
            "rmm": rmm_tokenize,
            "bmm": bmm_tokenize,
            "hmm": HmmTokenizer().tokenize,
        }

        self.pos_tagger = HmmPosTagger()

        self.constituency_parser = CKYParser(
            *read_grammar(grammar="""
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
            """),
            root_value="S"
        )

    def tokenize(self, text: str, tokenizer: str = "hmm") -> List[str]:
        """Text tokenization.

        :param text: (str) text to tokenize.
        :param tokenizer: (str) name of the tokenizer.
        :return: (List[str]) tokens of the text.
        """

        tokenizer = self.tokenize_funcs[tokenizer]
        return tokenizer(text)

    def pos_tag(self, tokens: List[str]) -> List[str]:
        """Part-of-speech tagging.

        :param tokens: (List[str]) Tokens for pos tagging.
        :return: (List[str]) POS labels for the tokens.
        """

        return self.pos_tagger.predict(tokens)

    def constituency_parse(self, tokens: List[str]):
        """Constituency syntax parsing.

        :param tokens: (List[str]): tokens to parse.
        :return: (Node): root node of the parsed tree.
        """

        return self.constituency_parser.parse(tokens=tokens)
