from statistical_chinese_parser import StatisticalChineseParser

if __name__ == '__main__':
    text = "北京和上海举行新年晚会"

    parser = StatisticalChineseParser()
    tokens = parser.tokenize(text=text, tokenizer="hmm")
    constituency_root = parser.constituency_parse(tokens=tokens)

    print(constituency_root)
