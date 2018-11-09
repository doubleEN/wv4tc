#!/usr/bin/env python
# -*- coding:utf-8 -*-

# author: jxm_2@hotmail.com
# date: 2018/11/7
# desc: 评估模型

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from wv4tc.feature_text import get_bow, stop_words
import pickle
import os
import jieba
import logging

log_file = "../log/easenet_test.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
    datefmt='%Y-%m-%d %A %H:%M:%S',
    filename=log_file,
    filemode='a'
)


def save_model(model, transfer, pkl_path):
    """
    模型序列化
    :param model:模型对象
    :param transfer: CountVec对象
    :param pkl_path: 序列化路径，以/结尾
    """
    if not os.path.exists(pkl_path):
        logging.error("路径不存在")
        return
    m_path = pkl_path + "model.pkl"
    transfer_path = pkl_path + "transfer.pkl"

    logging.info("序列化... ")
    with open(m_path, "wb") as fw:
        pickle.dump(model, fw)
    with open(transfer_path, "wb") as fw:
        pickle.dump(transfer, fw)
    logging.info("序列化完成")


def load_data(data_path, decoding="utf8"):
    """
    加载语料，语料一行一个样本，分割为 content，label两个字段
    :return: DataFrame对象
    """
    return pd.read_table(data_path, sep="\t", header=None, names=["content", "label"], encoding=decoding)


def eval_model(train_data_path, test_data_path, model, pkl_path):
    """
    模型测试
    """
    logging.info("TEST function.")

    logging.info("<load " + train_data_path + " and " + test_data_path + ">")
    train_data = load_data(train_data_path)
    test_data = load_data(test_data_path)
    boundary = len(train_data)
    all_data = pd.concat([train_data, test_data])
    doc_term, data_X, data_Y = get_bow(all_data, stop_words, content_name="content", label_name="label")

    logging.info(type(model))
    logging.info("boundary:" + str(boundary))

    train_X = data_X[:boundary]
    train_Y = data_Y[:boundary]
    test_X = data_X[boundary:]
    test_Y = data_Y[boundary:]

    model.fit(train_X, train_Y)

    if pkl_path:
        save_model(model, doc_term, pkl_path)

    y_true, y_pred = test_Y, model.predict(test_X)
    res = classification_report(y_true, y_pred)
    logging.info(res)
    logging.info("<<<[tuning end]>>>\n")


def load_model(model_path):
    """
    模型反序列化
    :param model_path: 模型文件夹路径
    :return: 模型，CountVec
    """
    if not model_path.endswith("/") or not model_path.endswith("\\"):
        model_path += "/"
    m_path = model_path + "model.pkl"
    transfer_path = model_path + "transfer.pkl"
    if not os.path.exists(m_path) or not os.path.exists(transfer_path):
        logging.error("模型路径非法")
        return
    with open(m_path, "rb") as fr:
        model = pickle.load(fr)
    with open(transfer_path, "rb") as fr:
        transfer = pickle.load(fr)
    return model, transfer


def predict(docs, model_path, model=None, transfer=None):
    """
    指定模型进行预测
    :param doc: 已分词文本数组
    :param model: 模型
    :param transfer: CountVec
    :param model_path: 序列化模型文件夹路径，默认不加载
    :return: label_pred
    """
    if not model_path:
        return model.predict(transfer.transform(docs))
    else:
        model, transfer = load_model(model_path)
        return model.predict(transfer.transform(docs))


def seg_predict(doc, model_path, model=None, transfer=None):
    """
    对字符串文本分词并预测
    :param doc:a str or a list，待预测的文本
    """
    if isinstance(doc, str):
        doc_arr = [" ".join(jieba.lcut(doc))]
        return predict(doc_arr, model_path, model=None, transfer=None)
    if isinstance(doc, list):
        docs_seg = [" ".join(jieba.lcut(d)) for d in doc]
        return predict(docs_seg, model_path, model=None, transfer=None)
