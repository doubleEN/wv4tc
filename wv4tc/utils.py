#!/usr/bin/env python
# -*- coding:utf-8 -*-

# author:jxm_2@hotmail.com
# datetime:2018/11/3

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
