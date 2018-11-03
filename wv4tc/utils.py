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
