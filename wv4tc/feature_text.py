#!/usr/bin/env python
# -*- coding:utf-8 -*-

# author:jxm_2@hotmail.com
# date:2018/11/5
# desc:文本特征处理

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
import numpy as np
from wv4tc.w2v import word2vec


def get_bow(data, stopwords, content_name, label_name):
    """
    加载文本语料，获得doc-term矩阵。
    保留单字符但去掉停用词。
    按指定的有序列名加载train（与label）。
    """
    count_vec = CountVectorizer(
        stop_words=stopwords,
        token_pattern="\w+"
    )

    X_mat = count_vec.fit_transform(data[content_name])
    if label_name is None:
        return count_vec, X_mat
    return count_vec, X_mat, data[label_name]


def get_mean_vec(data, w2v, stopwords, content_name, label_name, pkl_path):
    """
    获得文本集的平均向量
    :param data: 文本集
    :param w2v: word2vec对象
    :param stopwords: 停用词
    :param content_name: 训练文本
    :param label_name: 训练label
    """
    if not isinstance(w2v, word2vec):
        print("非法向量词典")
        return
    contents = data[content_name]
    data_vec = [w2v.mean_vect_sent(content, " ", stopwords) for content in contents]
    data_vec = np.array(data_vec)
    if pkl_path is not None:
        with open(pkl_path, "wb") as fw:
            labels = data[label_name].values
            labels = np.reshape(labels, (len(labels), 1))
            pickle.dump(np.append(data_vec, labels, axis=1))
    return w2v, data_vec, data[label_name]


def get_sum_vec(data, w2v, stopwords, content_name, label_name, pkl_path):
    """
    获得文本集的和向量
    :param data: 文本集
    :param w2v: word2vec对象
    :param stopwords: 停用词
    :param content_name: 训练文本
    :param label_name: 训练label
    """
    if not isinstance(w2v, word2vec):
        print("非法向量词典")
        return
    contents = data[content_name]
    data_vec = [w2v.sum_vect_sent(content, " ", stopwords) for content in contents]
    data_vec = np.array(data_vec)
    if pkl_path is not None:
        with open(pkl_path, "wb") as fw:
            labels = data[label_name].values
            labels = np.reshape(labels, (len(labels), 1))
            pickle.dump(np.append(data_vec, labels, axis=1))
    return w2v, np.array(data_vec), data[label_name]


def get_tfidf(data, stopwords, content_name, label_name):
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

    X_mat = count_vec.fit_transform(data[content_name])

    tfidf_transformer = TfidfTransformer(
        smooth_idf=True,
    ).fit(X_mat)
    X_train_tf = tfidf_transformer.transform(X_mat)

    if label_name is None:
        return count_vec, X_train_tf
    return count_vec, X_train_tf, data[label_name]
