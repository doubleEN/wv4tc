#!/usr/bin/env python
# -*- coding:utf-8 -*-

# author: jxm_2@hotmail.com
# date: 2018/11/7
# desc: 评估模型

import pandas as pd
from sklearn.metrics import *
from wv4tc.feature_text import get_bow
from wv4tc.utils import load_data, sbc2dbc, load_vecs
import pickle
import os
import jieba
import logging

# log_file = "../log/test_easenet_vec.log"
# log_file = "../log/test_fudan_vec.log"
log_file = "../log/test_sohu_vec.log"


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
    datefmt='%Y-%m-%d %A %H:%M:%S',
    filename=log_file,
    filemode='a'
)


def save_model(pkl_path, model, transfer=None):
    """
    模型序列化
    :param model:模型对象
    :param transfer: CountVec对象
    :param pkl_path: 序列化路径，以/结尾
    """
    if not os.path.exists(pkl_path):
        print("路径不存在")
        return
    if not pkl_path.endswith("/") or not pkl_path.endswith("\\"):
        pkl_path += "/"

    m_path = pkl_path + "model.pkl"
    print("序列化... ")
    with open(m_path, "wb") as fw:
        pickle.dump(model, fw)
    if transfer is not None:
        transfer_path = pkl_path + "transfer.pkl"
        with open(transfer_path, "wb") as fw:
            pickle.dump(transfer, fw)
    print("序列化完成")


def eval_model_vec(train_data, test_data, model, pkl_path):
    logging.info("---------------------------cut_off_begin--------------------------------")
    logging.info("VEC TESTING function.")

    if isinstance(train_data, str) and isinstance(test_data, str):
        logging.info("<load " + train_data + " and " + test_data + ">")
        train_X, train_Y = load_vecs(train_data)
        test_X, test_Y = load_vecs(test_data)
    else:
        train_X, train_Y = train_data
        test_X, test_Y = test_data
        logging.info("<loading train data size:" + str(train_X.shape) + " and loading test data size:" + str(
            test_X.shape) + ">")

    logging.info(type(model))
    logging.info("boundary:" + str(len(train_X)))

    logging.info("building...")
    model.fit(train_X, train_Y)
    logging.info("building finished.")

    if pkl_path is not None:
        save_model(pkl_path, model)
    logging.info("pkl finished.")

    y_true, y_pred = test_Y, model.predict(test_X)
    res = classification_report(y_true, y_pred)
    logging.info(res)
    res = precision_recall_fscore_support(y_true, y_pred)
    logging.info(res)
    res = confusion_matrix(y_true, y_pred)
    logging.info(res)
    logging.info("<<<[tuning end]>>>\n")
    return res


def eval_model(train_data, test_data, model, pkl_path, stop_words):
    """
    模型测试
    """
    logging.info("TEST function.")
    if isinstance(train_data, str) and isinstance(test_data, str):
        logging.info("<load " + train_data + " and " + test_data + ">")
        train_data = load_data(train_data)
        test_data = load_data(test_data)
    else:
        logging.info("<loading train data size:" + str(train_data.shape) + " and loading test data size:" + str(
            test_data.shape) + ">")
    boundary = len(train_data)
    all_data = pd.concat([train_data, test_data])
    doc_term, data_X, data_Y = get_bow(all_data, stop_words, content_name="content", label_name="label")

    logging.info(type(model))
    logging.info("boundary:" + str(boundary))

    train_X = data_X[:boundary]
    train_Y = data_Y[:boundary]
    test_X = data_X[boundary:]
    test_Y = data_Y[boundary:]

    logging.info("building...")
    model.fit(train_X, train_Y)
    logging.info("building finished.")

    if pkl_path is not None:
        save_model(pkl_path, model, doc_term)

    y_true, y_pred = test_Y, model.predict(test_X)
    res = classification_report(y_true, y_pred)
    logging.info(res)
    res = precision_recall_fscore_support(y_true, y_pred)
    logging.info(res)
    res = confusion_matrix(y_true, y_pred)
    logging.info(res)
    logging.info("<<<[tuning end]>>>\n")
    return res


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
        print("模型路径非法")
        return
    with open(m_path, "rb") as fr:
        model = pickle.load(fr)
    with open(transfer_path, "rb") as fr:
        transfer = pickle.load(fr)
    return model, transfer


def load_model_vec(model_path):
    """
    模型反序列化
    :param model_path: 模型文件路径
    :return: 模型
    """
    if not os.path.exists(model_path):
        print("模型路径非法")
        return
    with open(model_path, "rb") as fr:
        model = pickle.load(fr)
    return model


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
    doc = sbc2dbc(doc)
    if isinstance(doc, str):
        doc_arr = [" ".join(jieba.lcut(doc))]
        return predict(doc_arr, model_path, model=None, transfer=None)
    if isinstance(doc, list):
        docs_seg = [" ".join(jieba.lcut(d)) for d in doc]
        return predict(docs_seg, model_path, model=None, transfer=None)
