#!/usr/bin/env python
# -*- coding:utf-8 -*-

# author:jxm_2@hotmail.com
# datetime:2018/11/1


import os
import re
import logging
from wv4tc.utils import *

logging.basicConfig(level=logging.ERROR)
sohu_invalid_count = 0
sohu_sum = 0
valid_count = 0


def clean_text(text):
    """
    清理文本中的制表符、换行符，全角转半角，去掉特殊字符。
    """
    text = text.replace("\t", " ")
    text = text.replace("\n", "")
    sc_set = load_SC()
    for c in sc_set:
        text = text.replace(c, "")
    return sbc2dbc(text)


def doc_fudan(corpus_path, encoding="gb18030"):
    """
    fudan语料目录解析函数
    :param corpus_path:预料路径
    :param encoding:解码
    :return:str(content\tlabel)
    """
    file_names = os.listdir(corpus_path)
    for file in file_names:
        logging.debug("加载文档目录 " + file)
        label = file[file.index("-") + 1:]
        for doc in os.listdir(corpus_path + "/" + file):
            text = open(corpus_path + "/" + file + "/" + doc, "r", encoding=encoding, errors="ignore").read().strip()
            yield clean_text(text) + "\t" + label


def doc_easenet(corpus_path, encoding="utf8"):
    """
    网易文本分类语料解析
    :param corpus_path:corpus路径
    :param encoding:解码
    :return:content\tlabel
    """
    if not os.path.exists(corpus_path):
        logging.error("corpus_path 无效")
        return
    file_names = os.listdir(corpus_path)
    for file in file_names:
        label = file[:file.index("_")]
        text = open(corpus_path + "/" + file, "r", encoding=encoding, errors="ignore").read().strip()
        logging.debug(file + " 加载完成")
        yield clean_text(text) + "\t" + label


def parse_sohu_xml(xml_str):
    """
    解析一个搜狐xml形式的doc
    :param xml_str:doc
    :return:title\tcontent\tlabel
    """
    global sohu_invalid_count, sohu_sum
    sohu_sum += 1
    label_match = re.search("//(.*?)\\.sohu\\.com", xml_str)
    if label_match is None or label_match.group(1).strip() == "":
        logging.info(xml_str + "#### label 无法被解析")
        sohu_invalid_count += 1
        return
    label = label_match.group(1).strip()

    title_match = re.search("<contenttitle>(.*?)</contenttitle>", xml_str)
    if title_match is None or title_match.group(1).strip() == "":
        logging.info(xml_str + "#### title无法被解析")
        sohu_invalid_count += 1
        return
    title = title_match.group(1).strip()

    content_match = re.search("<content>(.*?)</content>", xml_str)
    if content_match is None or content_match.group(1).strip() == "":
        logging.info(xml_str + "#### content 无法被解析")
        sohu_invalid_count += 1
        return
    content = content_match.group(1).strip()
    return clean_text(title) + "\t" + clean_text(content) + "\t" + label


def doc_sohu(corpus_path, encoding="gb18030", parser=parse_sohu_xml):
    """
    搜狐xml文本解析
    :param corpus_path:corpus路径
    :param encoding:解码
    :param parser:xml解析器
    :return:去xml格式的纯文本内容
    """
    doc_xml = ""
    for line in open(corpus_path, "r", encoding=encoding):
        line = line.strip()
        if line == "</doc>":
            doc_xml += line
            doc = parser(doc_xml)
            doc_xml = ""
            if doc is not None:
                yield doc
        else:
            doc_xml += line


def format_doc(corpus_generator, corpus_path, out_path, r_encoding, w_encoding="utf-8", w_mode="w"):
    """
    加载原始语料到一个文本中，一行一个文本，制表符分割，label为最后一列
    :param corpus_generator: 语料访问生成器
    :param corpus_path:语料路径
    :param out_path:输入路径
    :param r_encoding:解码方式
    :param w_encoding:编码方式
    :param w_mode:写模式
    """
    if out_path and not w_encoding or not out_path and w_encoding:
        logging.error("参数 out_path、w_encoding 形式不合法。")
        return
    global valid_count
    with open(out_path, mode=w_mode, encoding=w_encoding) as fw:
        for doc in corpus_generator(corpus_path, r_encoding):
            fw.write(doc + "\n")
            valid_count += 1
    logging.debug("有效格式化文本数：" + str(valid_count))


