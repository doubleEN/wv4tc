#!/usr/bin/env python
# -*- coding:utf-8 -*-

# author: jxm_2@hotmail.com
# date: 2018/11/7
# desc: word2vec加载

import numpy as np


class word2vec():
    def __init__(self, w2v_path, dim, decoding="utf8"):
        '''
        加载word2vec类型
        :param w2v_path:w2v文本路径
        :param dim: 向量维度
        :param decoding: 解码
        '''
        self.w2v_dict = {}
        self.dim = dim
        for line in open(w2v_path, "r", encoding=decoding):
            parts = line.strip().split(" ")
            if len(parts) != dim + 1:
                print(line.strip() + "不能解析为一个向量对")
                continue
            self.w2v_dict[parts[0]] = np.array(parts[1:]).astype(np.float64)
        self.word_dict = self.w2v_dict.keys()

    def get_vec(self, word):
        """
        查找词的向量
        """
        if word not in self.word_dict:
            print(word + " 不存在于向量词典中")
            return np.zeros(shape=(self.dim))
        return self.w2v_dict[word]

    def sum_vect_sent(self, sentence, sep=" ", stopwords=[]):
        """
        词向量求和得句子向量
        :param sentence: 句子字符串
        :param sep: 句子分隔符
        :return: (句子向量，停用词命中数)
        """
        if sentence == None or sentence.strip() == "":
            return np.zeros(shape=(self.dim))
        sum_vec = 0.0
        tokens = sentence.strip().split(sep)
        stop_count = 0
        for token in tokens:
            if token.strip() == "":
                continue
            if token not in stopwords:
                sum_vec += self.get_vec(token)
            else:
                stop_count += 1
                # print("过滤停用词：" + token)
        if isinstance(sum_vec, float):
            return np.zeros(shape=(self.dim)), stop_count
        return sum_vec, stop_count

    def mean_vect_sent(self, sentence, sep=" ", stopwords=[]):
        """
        词向量求均值得句子向量
        :param sentence: 句子字符串
        :param sep: 句子分隔符
        :return: 句子向量
        """
        sum_vec, stop_count = self.sum_vect_sent(sentence, sep, stopwords)
        if (len(sentence.strip().split(sep)) - stop_count) == 0:
            print(sentence + " >>> 已分割过滤")
            return np.zeros(shape=(self.dim))
        return sum_vec / (len(sentence.strip().split(sep)) - stop_count)
