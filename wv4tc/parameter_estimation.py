#!/usr/bin/env python
# -*- coding:utf-8 -*-

# author:jxm_2@hotmail.com
# date:2018/11/6
# desc:调参并测试

import logging
from wv4tc.feature_text import get_bow, stop_words
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *


def tuning(train_path, test_path, model, tuned_parameters, scores, cv=5, n_jobs=-1, log_name="default.log"):
    logging.basicConfig(filename="../log/" + log_name, filemode="a", level=logging.INFO)

    logging.info("TUNING function.")
    train_data = pd.read_table(train_path, sep="\t", header=None, names=["content", "label"])
    test_data = pd.read_table(test_path, sep="\t", header=None, names=["content", "label"])
    all_data = pd.concat([train_data, test_data])
    doc_term, data_X, data_Y = get_bow(all_data, stop_words, content_name="content", label_name="label")
    boundary = len(train_data)

    logging.info("<load " + train_path + " and " + test_path + ">")
    logging.info("cv:" + str(cv))
    logging.info("tuned_parameters:" + str(tuned_parameters))
    logging.info(type(model))
    logging.info("boundary:" + str(boundary))

    train_X = data_X[:boundary]
    train_Y = data_Y[:boundary]
    test_X = data_X[boundary:]
    test_Y = data_Y[boundary:]

    for score in scores:
        logging.info("[tuning begin]" + " score with <" + score + ">")
        clf = GridSearchCV(
            estimator=model,
            param_grid=tuned_parameters,
            cv=cv,
            scoring=score,
            n_jobs=n_jobs
        )

        logging.info("fit...")
        clf.fit(train_X, train_Y)
        logging.info("fit end")
        logging.info("best paras " + str(clf.best_params_))
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            logging.info("%0.9f  >>>  %r" % (mean, params))

        logging.info("GridSearchCV test...")
        y_true, y_pred = test_Y, clf.predict(test_X)

        logging.info(precision_recall_fscore_support(y_true, y_pred))
        logging.info(confusion_matrix(y_true, y_pred))
        logging.info(classification_report(y_true, y_pred))

        logging.info("<<<[tuning end]>>>\n")
