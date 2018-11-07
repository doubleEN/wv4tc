#!/usr/bin/env python
# -*- coding:utf-8 -*-

# author: jxm_2@hotmail.com
# date: 2018/11/7
# desc: word2vec加载

import logging
import numpy as np


class w2v():
    def __init__(self, w2v_path, dim, decoding="utf8"):
        '''
        加载word2vec类型
        :param w2v_path:w2v文本路径
        :param dim: 向量维度
        :param decoding: 解码
        '''
        self.w2v_dict = {}
        for line in open(w2v_path, "r", encoding=decoding):
            parts = line.strip().split(" ")
            if len(parts) != dim + 1:
                logging.warning(line.strip() + "不能解析为一个向量对")
                continue
            self.w2v_dict[parts[0]] = np.array(parts[1:]).astype(np.float32)
        self.word_dict = self.w2v_dict.keys()

    def get_vec(self, word):
        """
        查找词的向量
        """
        if word not in self.word_dict:
            print(word + " 不存在于向量词典中")
            return
        return self.w2v_dict[word]
