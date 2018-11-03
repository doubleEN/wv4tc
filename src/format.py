# -*- coding: UTF-8 -*-
import os
import re
import xml.etree.ElementTree as ET
import logging

logging.basicConfig(level=logging.DEBUG)
sohu_invalid_count = 0
sohu_incomplete_count = 0
url_set=set()


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
            yield text + "\t" + label


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
        yield text + "\t" + label


def parse_sohu_xml(xml_str):
    """
    解析一个搜狐xml形式的doc
    :param xml_str:doc
    :return:title\ncontent\tlabel
    """
    global sohu_invalid_count, sohu_incomplete_count
    try:
        root = ET.fromstring(xml_str)
    except Exception:
        sohu_invalid_count += 1
        logging.error(xml_str + "<>不能被正确解析.1")
        return
    url = root.findall("url")
    contenttitle = root.findall("contenttitle")
    content = root.findall("content")
    if len(url) != 1 or len(contenttitle) != 1 or len(content) != 1 or content[0].text is None or url[0].text is None or \
            contenttitle[0].text is None:
        logging.info(xml_str + "<>字段不完整")
        sohu_incomplete_count += 1
        return
    matches = re.search("//(.*?)\\.sohu\\.com", url[0].text)
    url_match = re.search("^(.*?)\\.com", url[0].text)
    url_set.add(url_match[1])
    label = matches[1]
    return contenttitle[0].text + "\n" + content[0].text + "\t" + label


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


def format_doc(corpus_generator, corpus_path, out_path, r_encoding, w_encoding="utf-8"):
    """
    加载原始语料成固定的文件目录格式[content\tlabel]
    :param corpus_generator: 语料访问生成器
    :param corpus_path:语料路径
    :param out_path:输入路径
    :param r_encoding:解码方式
    :param w_encoding:编码方式
    """
    if out_path and not w_encoding or not out_path and w_encoding:
        logging.error("参数 out_path、w_encoding 形式不合法。")
        return

    # 创建输出目录
    if os.path.exists(out_path):
        os.removedirs(out_path)
    os.mkdir(out_path)

    text_sum = 1
    for doc in corpus_generator(corpus_path, r_encoding):
        with open(out_path + "/" + str(text_sum) + ".txt", "w", encoding=w_encoding) as fw:
            fw.write(doc)
        text_sum += 1

    logging.debug(corpus_path + "格式化完成，共计文本数" + str(text_sum - 1))

