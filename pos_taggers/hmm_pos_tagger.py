import json
import math
import os
from typing import List

import pandas as pd


class HmmPosTagger:
    def __init__(self):
        self.start_prop = {}
        self.trans_prop = {}
        self.emit_prop = {}
        self.poslist = []
        self.trans_sum = {}
        self.emit_sum = {}

        fp_params = "data/hmm_pos_tagger.params"
        if os.path.exists(fp_params):
            with open(fp_params, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.start_prop = d["start_prop"]
            self.trans_prop = d["trans_prop"]
            self.emit_prop = d["emit_prop"]
            self.poslist = d["poslist"]
            self.trans_sum = d["trans_sum"]
            self.emit_sum = d["emit_sum"]
        else:
            self.train('data/199801.txt', fp_save=fp_params)

    def __upd_trans(self, curpos: str, nxtpos: str):
        """更新转移概率矩阵。

        Args:
            curpos (str): 当前词性。
            nxtpos (str): 下一词性。
        """

        if curpos in self.trans_prop:
            if nxtpos in self.trans_prop[curpos]:
                self.trans_prop[curpos][nxtpos] += 1
            else:
                self.trans_prop[curpos][nxtpos] = 1
        else:
            self.trans_prop[curpos] = {nxtpos: 1}

    def __upd_emit(self, pos: str, word: str):
        """更新发射概率矩阵。

        Args:
            pos (str): 词性。
            word (str): 词语。
        """

        if pos in self.emit_prop:
            if word in self.emit_prop[pos]:
                self.emit_prop[pos][word] += 1
            else:
                self.emit_prop[pos][word] = 1
        else:
            self.emit_prop[pos] = {word: 1}

    def __upd_start(self, pos: str):
        """更新初始状态矩阵。

        Args:
            pos (str): 初始词语的词性。
        """

        if pos in self.start_prop:
            self.start_prop[pos] += 1
        else:
            self.start_prop[pos] = 1

    def train(self, data_path: str, fp_save: str):
        """训练 hmm 模型、求得转移矩阵、发射矩阵、初始状态矩阵。

        Args:
            data_path (str): 训练数据的路径。
            fp_save (str): 参数保存路径。
        """

        f = open(data_path, 'r', encoding="utf-8")
        lines = f.read().strip().splitlines()
        for line in lines:
            line = line.strip().split()
            if not line:
                continue
            # 统计初始状态的概率
            self.__upd_start(line[0].split('/')[1])
            # 统计转移概率、发射概率
            for i in range(len(line) - 1):
                self.__upd_emit(line[i].split('/')[1], line[i].split('/')[0])
                self.__upd_trans(line[i].split('/')[1],
                                 line[i + 1].split('/')[1])
            i = len(line) - 1
            self.__upd_emit(line[i].split('/')[1], line[i].split('/')[0])
        f.close()
        # 记录所有的 pos
        self.poslist = list(self.emit_prop.keys())
        self.poslist.sort()
        # 统计 trans、emit 矩阵中各个 pos 的归一化分母
        num_trans = [
            sum(self.trans_prop[key].values()) for key in self.trans_prop
        ]
        self.trans_sum = dict(zip(self.trans_prop.keys(), num_trans))
        num_emit = [
            sum(self.emit_prop[key].values()) for key in self.emit_prop
        ]
        self.emit_sum = dict(zip(self.emit_prop.keys(), num_emit))

        # 保存参数
        parameters = {
            "start_prop": self.start_prop,
            "trans_prop": self.trans_prop,
            "emit_prop": self.emit_prop,
            "poslist": self.poslist,
            "trans_sum": self.trans_sum,
            "emit_sum": self.emit_sum,
        }
        jsonstr = json.dumps(parameters, ensure_ascii=False, indent=4)  # 输出上面的参数  输出到文件
        with open(fp_save, 'w', encoding='utf8') as jsonfile:
            jsonfile.write(jsonstr)

    def predict(self, tokens: List[str]):
        """Viterbi 算法预测词性。

        Args:
            tokens (List[str]): 词语序列。

        Returns:
            list: 词性标注序列。
        """

        posnum = len(self.poslist)
        dp = pd.DataFrame(index=self.poslist)
        path = pd.DataFrame(index=self.poslist)
        # 初始化 dp 矩阵（DP 矩阵: posnum * wordsnum 存储每个 word 每个 pos 的最大概率）
        start = []
        num_sentence = sum(self.start_prop.values()) + posnum
        for pos in self.poslist:
            sta_pos = self.start_prop.get(pos, 1e-16) / num_sentence
            sta_pos *= (self.emit_prop[pos].get(tokens[0], 1e-16) /
                        self.emit_sum[pos])
            sta_pos = math.log(sta_pos)
            start.append(sta_pos)
        dp[0] = start
        # 初始化 path 矩阵
        path[0] = ['_start_'] * posnum
        # 递推
        for t in range(1, len(tokens)):  # 句子中第 t 个词
            prob_pos, path_point = [], []
            for i in self.poslist:  # i 为当前词的 pos
                max_prob, last_point = float('-inf'), ''
                emit = math.log(self.emit_prop[i].get(tokens[t], 1e-16) / self.emit_sum[i])
                for j in self.poslist:  # j 为上一次的 pos
                    tmp = dp.loc[j, t - 1] + emit
                    tmp += math.log(self.trans_prop[j].get(i, 1e-16) / self.trans_sum[j])
                    if tmp > max_prob:
                        max_prob, last_point = tmp, j
                prob_pos.append(max_prob)
                path_point.append(last_point)
            dp[t], path[t] = prob_pos, path_point
        # 回溯
        prob_list = list(dp[len(tokens) - 1])
        cur_pos = self.poslist[prob_list.index(max(prob_list))]
        path_que = []
        path_que.append(cur_pos)
        for i in range(len(tokens) - 1, 0, -1):
            cur_pos = path[i].loc[cur_pos]
            path_que.append(cur_pos)
        # 返回结果
        postag = []
        for i in range(len(tokens)):
            postag.append(path_que[-i - 1])
        return postag


if __name__ == "__main__":
    hmm = HmmPosTagger()
    pos_list = hmm.predict(
        ["在", "这", "一", "年", "中", "，", "中国", "的", "改革", "开放", "和", "现代化", "建设", "继续", "向前", "迈进", "再次", "获得", "好", "的",
         "收成"])
    print(pos_list)

# 1. 语料库中有 26 个基本词类标记
#       形容词a、区别词b、连词c、副词d、叹词e、方位词f、语素g、前接成分h、成语i、
#       简称j、后接成分k、习惯用语l、数词m、名词n、拟声词o、介词p、量词q、代词r、
#       处所词s、时间词t、助词u、动词v、标点符号w、非语素字x、语气词y、状态词z、
#
#
# 2. 语料库中还有 74 个扩充标记：对于语素，具体区分为 Ag Bg Dg Mg Ng Rg Tg Vg Yg
#
#
# 3. 词性标注只标注基本词性，因此在数据清洗的过程中，将扩充标记归类到各个基本词类中，语素也归类到相应词类中
