#!/usr/bin/env python
# -*- coding:utf-8 -*-

# author:jxm_2@hotmail.com
# datetime:2018/11/3

import pandas as pd
import numpy as np


def sbc2dbc(str):
    """
    全角转半角
    """
    rstr = ""
    for uchar in str:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif inside_code >= 65281 and inside_code <= 65374:  # 全角字符根据关系转化
            inside_code -= 65248
        rstr += chr(inside_code)
    return rstr


def dbc2sbc(str):
    """
    半角转全角
    """
    rstr = ""
    for uchar in str:
        inside_code = ord(uchar)
        if inside_code == 32:  # 半角空格直接转化
            inside_code = 12288
        elif inside_code >= 32 and inside_code <= 126:  # 半角字符根据关系转化
            inside_code += 65248
        rstr += chr(inside_code)
    return rstr


def load_SC(path="../data/SpecialCharacter"):
    """
    加载特殊字符集
    """
    sc_set = set()
    for c in open(path, "r", encoding="utf8"):
        sc_set.add(c.strip())
    return sc_set


def load_dict(dict_path, decoding="utf8"):
    """
    加载文本词典，文本词典一行一个词，得到词集。
    """
    return {word.strip() for word in open(dict_path, "r", encoding=decoding) if not word.startswith("#")}


def load_data(data_path, decoding="utf8"):
    """
    加载语料，语料一行一个样本，分割为 content，label两个字段
    :return: DataFrame对象
    """
    return pd.read_table(data_path, sep="\t", header=None, names=["content", "label"], encoding=decoding)


def load_vecs(vec_path, decoding="utf8"):
    """
    加载文本向量集，格式：vec\tlabel。向量值单空格分割
    """
    data_set = []
    labels = []
    for line in open(vec_path, "r", encoding=decoding):
        line = line.strip()
        if line == "":
            continue
        data, label = line.split("\t")
        data = list(np.array(data.split(" ")).astype(np.float64))
        data_set.append(data)
        labels.append(label)
    return np.array(data_set), np.array(labels)


if __name__ == "__main__":
    x, y = load_vecs("../data/texts_vec")
    print(x.shape)
    print(y.shape)
