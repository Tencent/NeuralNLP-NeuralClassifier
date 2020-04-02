# coding=utf8


import re
import jieba
import json


money_pat = re.compile(r"[\d\.]+[亿万千百十个元]+")
time_pat_arr = [
    re.compile(r"[\d]+[年月日天小时]+"),
]

stop_words = list('.!?,\'/()，。《》|？（）【】 \t\n的上下“”了： ！、在我:也＂｜；❤') + ["建行",  "支行", "银行", "浙商", "中信"]


class NegWordPredictor(object):
    def __init__(self, f_path):
        """
        :param f_path: 构建的敏感词词库的地址
        """
        self.word_set = self._load_word(f_path)

    @staticmethod
    def _clean_text(text):
        clear_text = text
        clear_text = re.sub(money_pat, " ", clear_text)
        for time_pat in time_pat_arr:
            clear_text = re.sub(time_pat, " ", clear_text)
        return clear_text

    @staticmethod
    def _is_hit(word_arr, word_set):
        for word in word_arr:
            if word in word_set:
                return True
        return False

    @staticmethod
    def _load_word(word_path):
        word_set = set()
        with open(word_path) as in_:
            for line in in_:
                line_arr = line.strip().split()
                if len(line_arr) > 0:
                    word = line_arr[0]
                    word_set.add(word)
        return word_set

    def norm_text(self, text):
        text = self._clean_text(text)
        words = [word for word in jieba.cut(text) if word not in stop_words]
        return " ".join(words)

    def predict_sen(self, sen):
        sen = self.norm_text(sen)
        word_arr = sen.split()
        if self._is_hit(word_arr, self.word_set):
            return 1.0
        return 0.0

    def predict(self, sen_info):
        """
        :param sen_info 句子结构: {"doc_text": "中信银行“战疫”开工季: “有温度”的综合金融服务助力企业复工复产", "doc_label": ["正面"], "doc_token": ["中信银行", "“", "战疫", "”", "开工", "季", ":", " ", "“", "有", "温度", "”", "的", "综合", "金融服务", "助力", "企业", "复工", "复产"], "doc_keyword": [], "doc_topic": []}
        :return: 0： 表示句子不是负面； 1： 表示句子是负面
        """
        if isinstance(sen_info, str):
            sen_info = json.loads(sen_info)
        sen = sen_info.get("doc_text", None)
        if sen is None:
            return 0
        return self.predict_sen(sen)


if __name__ == "__main__":
    f_neg_words = "../data/rule_data/select_word.txt"
    test_predictor = NegWordPredictor(f_neg_words)
    test_sens = ["天雷滚滚 天津物产融资黑洞塌陷| 债市爆雷之十六", "浙商银行买入8亿元＂假理财”",
                 "被诈骗两年浑然不知 浙商银行买入8亿元“假理财”",
                 "“风电巨头”金风科技“钱紧”边卖资产边融资",
                 "建行支行行长虚构8亿“假理财”",
                 ]
    wr_cnt = 0
    for test_sen in test_sens:
        score = test_predictor.predict_sen(test_sen)
        print(test_sen, score)
        if score < 1.:
            wr_cnt += 1

    print(len(test_sens))
    print(wr_cnt)
