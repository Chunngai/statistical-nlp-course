from statistical_chinese_parser import StatisticalChineseParser

if __name__ == '__main__':
    text = "北京和上海举行新年晚会"

    parser = StatisticalChineseParser()

    # Tokenization.
    tokens = parser.tokenize(text=text, tokenizer="hmm")
    print(tokens)

    # Pos tagging.
    pos_list = parser.pos_tag(tokens=tokens)
    print(pos_list)

    # Syntax parsing.
    constituency_root = parser.constituency_parse(tokens=tokens)
    print(constituency_root)
