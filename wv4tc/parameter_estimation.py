#!/usr/bin/env python
# -*- coding:utf-8 -*-

# author:jxm_2@hotmail.com
# date:2018/11/6
# desc:调参并测试

import logging
from wv4tc.feature_text import get_bow
from wv4tc.utils import load_data, load_vecs
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

log_file = "../log/tuning_easenet_vec.log"
# log_file = "../log/tuning_ease.log"
# log_file = "../log/tuning_fudan_vec.log"
# log_file = "../log/tuning_sohu_vec.log"
# log_file = "../log/tuning_knn_vec.log"


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
    datefmt='%Y-%m-%d %A %H:%M:%S',
    filename=log_file,
    filemode='a'
)


def tuning_vec(train_data, test_data, model, tuned_parameters, scores, cv=5, n_jobs=-1):
    logging.info("---------------------------cut_off_begin--------------------------------")
    logging.info("VEC TUNING function.")

    if isinstance(train_data, str) and isinstance(test_data, str):
        logging.info("<load " + train_data + " and " + test_data + ">")
        train_X, train_Y = load_vecs(train_data)
        test_X, test_Y = load_vecs(test_data)
    else:
        train_X, train_Y = train_data
        test_X, test_Y = test_data
        logging.info("<loading train data size:" + str(train_X.shape) + " and loading test data size:" + str(
            test_X.shape) + ">")

    logging.info("cv:" + str(cv))
    logging.info("tuned_parameters:" + str(tuned_parameters))
    logging.info(type(model))
    logging.info("boundary:" + str(len(train_X)))

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
        logging.info("---------------------------cut_off_end--------------------------------")


def tuning(train_data, test_data, model, tuned_parameters, scores, stop_words, cv=5, n_jobs=-1):
    logging.info("---------------------------cut_off_begin--------------------------------")
    logging.info("TUNING function.")

    if isinstance(train_data, str) and isinstance(test_data, str):
        logging.info("<load " + train_data + " and " + test_data + ">")
        train_data = load_data(train_data, "utf8")
        test_data = load_data(test_data, "utf8")
    else:
        logging.info("<loading train data size:" + str(train_data.shape) + " and loading test data size:" + str(
            test_data.shape) + ">")
    all_data = pd.concat([train_data, test_data])
    doc_term, data_X, data_Y = get_bow(all_data, stop_words, content_name="content", label_name="label")
    boundary = len(train_data)

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
        logging.info("---------------------------cut_off_end--------------------------------")
