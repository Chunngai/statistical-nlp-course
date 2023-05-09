from statistical_chinese_parser import StatisticalChineseParser

if __name__ == '__main__':
    text = "北京和上海举行新年晚会"

    parser = StatisticalChineseParser()
    tokens = parser.tokenize(text=text, tokenizer="hmm")

    print(f"text: {text}")
    print(f"tokens: {tokens}")
