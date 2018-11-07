#!/usr/bin/env python
# -*- coding:utf-8 -*-

# author: jxm_2@hotmail.com
# date: 2018/11/7
# desc: 评估模型

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from wv4tc.feature_text import get_bow, stop_words

import logging
logging.basicConfig(filename="../log/test.log", filemode="a", level=logging.INFO)


def eval_model(train_data_path, test_data_path, model):
    logging.info("TEST function.")

    logging.info("<load " + train_data_path + " and " + test_data_path + ">")
    train_data = pd.read_table(train_data_path, sep="\t", header=None, names=["content", "label"])
    test_data = pd.read_table(test_data_path, sep="\t", header=None, names=["content", "label"])
    boundary = len(train_data)
    all_data = pd.concat([train_data, test_data])
    doc_term, data_X, data_Y = get_bow(all_data, stop_words, content_name="content", label_name="label")

    logging.info(type(model))
    logging.info("boundary:" + str(boundary))

    train_X = data_X[:boundary]
    train_Y = data_Y[:boundary]
    test_X = data_X[boundary:]
    test_Y = data_Y[boundary:]

    model.fit(train_X,train_Y)
    y_true, y_pred = test_Y, model.predict(test_X)
    logging.info(classification_report(y_true, y_pred))
    logging.info("<<<[tuning end]>>>\n")
