#!/usr/bin/env python
# -*- coding:utf-8 -*-

# author:jxm_2@hotmail.com
# date:2018/11/5
# desc:文本特征处理

from wv4tc.utils import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)

stop_words = load_dict("../data/stopwords.txt")


def get_bow(data_path, stopwords, names, content_name, label_name):
    """
    加载文本语料，获得doc-term矩阵。
    保留单字符但去掉停用词。
    按指定的有序列名加载train（与label）。
    """
    count_vec = CountVectorizer(
        stop_words=stopwords,
        token_pattern="\w+"
    )

    data = pd.read_table(data_path, sep="\t", header=None, names=names)
    X_mat = count_vec.fit_transform(data[content_name])
    logging.debug(data_path + " >>> BOW已加载")
    if label_name is None:
        return count_vec, X_mat
    return count_vec, X_mat, data[label_name]


def get_tfidf(data_path, stopwords, names, content_name, label_name):
    """
    加载文本语料为TF-IDF矩阵。
    保留单字符但去掉停用词。
    按指定的有序列名加载train（与label）。
    TF-IDF计算进行平滑。
    返回第一个元素为文本语料的NOW。
    """
    count_vec = CountVectorizer(
        stop_words=stopwords,
        token_pattern="\w+"
    )

    data = pd.read_table(
        data_path, sep="\t",
        header=None,
        names=names
    )
    X_mat = count_vec.fit_transform(data[content_name])
    logging.debug(data_path + " >>> BOW 已加载")

    tf_transformer = TfidfTransformer(
        smooth_idf=True,
    ).fit(X_mat)
    X_train_tf = tf_transformer.transform(X_mat)
    logging.debug(data_path + " >>> TF-IDF 已加载")

    if label_name is None:
        return count_vec, X_train_tf
    return count_vec, X_train_tf, data[label_name]

