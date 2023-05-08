from statisticalchineseparser import StatisticalChineseParser

if __name__ == '__main__':
    text = "一个基于统计的中文文本解析器。"

    parser = StatisticalChineseParser()

    # Tokenization.
    tokens = parser.tokenize(text=text, tokenizer="hmm")
    print(tokens)

    # Pos tagging.
    pos_list = parser.pos_tag(tokens=tokens)
    print(pos_list)

    # Syntax parsing.
    constituency_root = parser.constituency_parse(text=text)
    print(constituency_root)
