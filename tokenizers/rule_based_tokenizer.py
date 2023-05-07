import re

wordless = []
with open("tokenizers/data/wordless.txt", "r", encoding="UTF-8") as f:
    lines = f.read()
for word in lines:
    wordless.append(word)

dic = []
with open("tokenizers/data/199801.txt", "r", encoding="gbk") as f:
    lines = f.read()
r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~（）]+'
lines = re.sub(r1, '', lines)
lines = re.split(r" ", lines)
for word in lines:
    if word not in wordless and word != "\n" and word != "":
        dic.append(word)


def select_word(text):
    result = 0
    for i in text:
        if len(i) == 1:
            result += 1
    return result


def fully_tokenize(text):
    word_list = []
    for i in range(len(text)):
        for j in range(i + 1, len(text) + 1):
            word = text[i:j]
            if word in dic:
                word_list.append(word)
    return word_list


def mm_tokenize(text):
    word_list = []
    i = 0
    while i < len(text):
        longest_word = text[i]
        for j in range(i + 1, len(text) + 1):
            word = text[i:j]
            if word in dic:
                if len(word) > len(longest_word):
                    longest_word = word
        word_list.append(longest_word)
        i += len(longest_word)
    return word_list


def rmm_tokenize(text):
    word_list = []
    i = len(text) - 1
    while i >= 0:
        longest_word = text[i]
        for j in range(0, i):
            word = text[j:i + 1]
            if word in dic:
                if len(word) > len(longest_word):
                    longest_word = word
        word_list.insert(0, longest_word)
        i -= len(longest_word)
    return word_list


def bmm_tokenize(text):
    list_forward = mm_tokenize(text)
    list_backward = rmm_tokenize(text)
    if len(list_forward) > len(list_backward):
        list_final = list_backward[:]
    elif len(list_forward) < len(list_backward):
        list_final = list_forward[:]
    else:
        if select_word(list_forward) > select_word(list_backward):
            list_final = list_backward[:]
        elif select_word(list_forward) < select_word(list_backward):
            list_final = list_forward[:]
        else:
            list_final = list_backward[:]
    return list_final


if __name__ == "__main__":
    text = "这是一个句子。"

    mm_tokens = mm_tokenize(text)
    rmm_tokens = rmm_tokenize(text)
    bmm_tokens = bmm_tokenize(text)
    print("正向最长匹配", mm_tokens)
    print("逆向最长匹配", rmm_tokens)
    print("双向最长匹配", bmm_tokens)
