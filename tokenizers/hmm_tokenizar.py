# -*- encoding: utf-8 -*-
import json
import os

import pandas as pd


class HmmTokenizer:
    def __init__(self):
        self.start_p = {'S': 0, 'B': 0, 'M': 0, 'E': 0}
        self.trans_p = {'S': {}, 'B': {}, 'M': {}, 'E': {}}
        self.emit_p = {'S': {}, 'B': {}, 'M': {}, 'E': {}}
        self.state_num = {'S': 0, 'B': 0, 'M': 0, 'E': 0}
        self.state_list = ['S', 'B', 'M', 'E']
        self.line_num = 0
        self.smooth = 1e-6

        fp_params = "data/hmm_tokenizer.params"
        if os.path.exists(fp_params):
            with open(fp_params, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.start_p = d["start_p"]
            self.trans_p = d["trans_p"]
            self.emit_p = d["emit_p"]
        else:
            self.train('data/199801.txt', fp_save=fp_params)

    @staticmethod
    def __state(word: str):
        """获取词语的BOS标签。

        标注采用 4-tag 标注方法，
        tag = {S,B,M,E}，S表示单字为词，B表示词的首字，M表示词的中间字，E表示词的结尾字

        Args:
            word (str): 函数返回词语 word 的状态标签。
        """

        if len(word) == 1:
            state = ['S']
        else:
            state = list('B' + 'M' * (len(word) - 2) + 'E')

        return state

    def train(self, filepath: str, fp_save: str):
        """训练hmm, 学习发射概率、转移概率等参数。

        Args:
            filepath (str): 训练语料的路径。
            fp_save (str): 参数路径。
        """

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split()
                line = [
                    word.split("/")[0]
                    for word in line
                ]

                self.line_num += 1
                # 获取观测（字符）序列
                char_seq = list(''.join(line))
                # 获取状态（BMES）序列
                state_seq = []
                for word in line:
                    state_seq.extend(self.__state(word))
                # 判断是否等长
                assert len(char_seq) == len(state_seq)
                # 统计参数
                for i, s in enumerate(state_seq):
                    self.state_num[s] = self.state_num.get(s, 0) + 1.0
                    self.emit_p[s][char_seq[i]] = self.emit_p[s].get(
                        char_seq[i], 0) + 1.0
                    if i == 0:
                        self.start_p[s] += 1.0
                    else:
                        last_s = state_seq[i - 1]
                        self.trans_p[last_s][s] = self.trans_p[last_s].get(
                            s, 0) + 1.0
        # 归一化：
        self.start_p = {
            k: (v + 1.0) / (self.line_num + 4)
            for k, v in self.start_p.items()
        }
        self.emit_p = {
            k: {w: num / self.state_num[k]
                for w, num in dic.items()}
            for k, dic in self.emit_p.items()
        }
        self.trans_p = {
            k1: {k2: num / self.state_num[k1]
                 for k2, num in dic.items()}
            for k1, dic in self.trans_p.items()
        }

        # 保存参数
        parameters = {
            'start_p': self.start_p,
            'trans_p': self.trans_p,
            'emit_p': self.emit_p
        }
        jsonstr = json.dumps(parameters, ensure_ascii=False, indent=4)  # 输出上面的参数  输出到文件
        with open(fp_save, 'w', encoding='utf8') as jsonfile:
            jsonfile.write(jsonstr)

    def viterbi(self, text: str):
        """Viterbi 算法。

        Args:
            text (str): 句子。

        Returns:
            list: 最优标注序列。
        """

        text = list(text)
        dp = pd.DataFrame(index=self.state_list)
        # 初始化 dp 矩阵 (prop，last_state)
        dp[0] = [(self.start_p[s] * self.emit_p[s].get(text[0], self.smooth),
                  '_start_') for s in self.state_list]
        # 动态规划地更新 dp 矩阵
        for i, ch in enumerate(text[1:]):  # 遍历句子中的每个字符 ch
            dp_ch = []

            for s in self.state_list:  # 遍历当前字符的所有可能状态
                emit = self.emit_p[s].get(ch, self.smooth)
                # 遍历上一个字符的所有可能状态，寻找经过当前状态的最优路径
                (prob, last_state) = max([
                    (dp.loc[ls, i][0] * self.trans_p[ls].get(s, self.smooth) *
                     emit, ls) for ls in self.state_list
                ])
                dp_ch.append((prob, last_state))

            dp[i + 1] = dp_ch

        # 回溯最优路径
        path = []
        end = list(dp[len(text) - 1])
        back_point = self.state_list[end.index(max(end))]  # 找到最后一个字符的最大的状态，比如s，加到原来的path
        path.append(back_point)

        for i in range(len(text) - 1, 0, -1):
            back_point = dp.loc[back_point, i][1]  # 指所有字符的
            path.append(back_point)
        path.reverse()

        return path

    def tokenize(self, text: str):
        """根据 viterbi 算法获得状态，根据状态切分句子。

        Args:
            text (str): 待分词的句子。

        Returns:
            list: 分词列表。
        """

        state = self.viterbi(text)
        cut_res = []  # 切分
        begin = 0
        for i, ch in enumerate(text):
            if state[i] == 'B':
                begin = i
            elif state[i] == 'E':
                cut_res.append(text[begin:i + 1])
            elif state[i] == 'S':
                cut_res.append(text[i])
        return cut_res


if __name__ == "__main__":
    hmm = HmmTokenizer()
    tokens = hmm.tokenize('中央电视台收获一批好剧本，研究生命起源，欢迎新老师生过来')
    print(tokens)
