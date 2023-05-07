#读字典
import  re

wordless = []
f = open("./1.txt", "r", encoding="UTF-8")
lines = f.read()
for word in lines:
    wordless.append(word)


dict = []
f = open("./199801.txt", "r", encoding="gbk")
lines= f.read()
r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~（）]+'
r2 = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
r3 = "[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\\\\，。=？、：“”‘’《》【】￥……（）]+"
lines = re.sub(r1,'',lines)

lines= re.split(r" ",lines)
for word in lines:
    if(word not in wordless and word!="\n" and word !=""):
        dict.append(word)


def select_word(text):
    result = 0
    for i in text:
        if (len(i) == 1):
            result += 1
    return result


def fully_segment(text, dic):
    word_list = []
    for i in range(len(text)):
        for j in range(i + 1, len(text) + 1):
            word = text[i:j]
            if word in dic:
                word_list.append(word)
    return word_list


def forward_segment(text, dic):
    word_list = []
    i = 0
    while (i < len(text)):
        longest_word = text[i]
        for j in range(i + 1, len(text) + 1):
            word = text[i:j]
            if word in dic:
                if len(word) > len(longest_word):
                    longest_word = word
        word_list.append(longest_word)
        i += len(longest_word)
    return word_list


def backward_segment(text, dic):
    word_list = []
    i = len(text) - 1
    while (i >= 0):
        longest_word = text[i]
        for j in range(0, i):
            word = text[j:i + 1]
            if word in dic:
                if len(word) > len(longest_word):
                    longest_word = word
        word_list.insert(0, longest_word)
        i -= len(longest_word)
    return word_list


def all_segment(text, dic):
    list_forward = forward_segment(text, dic)
    list_backward = backward_segment(text, dic)
    list_final = []
    if (len(list_forward) > len(list_backward)):
        list_final = list_backward[:]
    elif (len(list_forward) < len(list_backward)):
        list_final = list_forward[:]
    else:
        if (select_word(list_forward) > select_word(list_backward)):
            list_final = list_backward[:]
        elif (select_word(list_forward) < select_word(list_backward)):
            list_final = list_forward[:]
        else:
            list_final = list_backward[:]
    return list_final


# dic = ["项目", "研究", "目的", "商品", "服务", "和服", "研究生", "结婚", "和尚", "结婚", "尚未", \
#        "生命", "起源", "当下", "雨天", "地面", "积水", "下雨天", "欢迎", "老师", "生前", "就餐", "迎新", "师生", "前来"]

if __name__ == "__main__":
    while (1):
        a = input("请输入你要分词的句子：（输入0结束输入）")
        if (a == '0'):
            print("输入结束！")
            break
        b = fully_segment(a, dict)
        print("分词的结果", b)
        list_forward = forward_segment(a, dict)
        list_backward = backward_segment(a, dict)
        list_all = all_segment(a, dict)
        print("正向最长匹配", list_forward)
        print("逆向最长匹配", list_backward)
        print("双向最长匹配", list_all)
