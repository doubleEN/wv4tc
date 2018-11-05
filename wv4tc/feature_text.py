#!/usr/bin/env python
# -*- coding:utf-8 -*-

# author:jxm_2@hotmail.com
# date:2018/11/5
# desc:文本特征处理

from wv4tc.utils import *
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

stop_words = load_dict("../data/stopwords.txt")

def get_bow(data_path, stopwords, names, content_name, label_name):
    """
    加载文本语料，获得doc-term矩阵。
    保留单字符但去掉停用词。
    按指定的有序列名加载train与label。
    """
    count_vec = CountVectorizer(
        stop_words=stopwords,
        token_pattern="\w+"
    )

    data = pd.read_table(data_path, sep="\t", header=None, names=names)
    X_mat = count_vec.fit_transform(data[content_name])
    return X_mat, data[label_name], count_vec

