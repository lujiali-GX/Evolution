# -*- coding: utf-8 -*-

"""
----
文件名称: model_metrics.py
----
\\
----
模块概述: 模型任务指标
----
\\
----
作   者: ljl (996153075@qq.com)
----
\\
----
创建日期: 2025/3/28
----
\\
----
版    本: 1.0.0
----
\\
----
依    赖:
----
    - sys
\\
\\
----
使用示例：
----
    - model_metrics.main()
\\
\\
----    
异常处理：
----
    -
\\
\\
----
注意事项：
----
    -
\\
"""

# ================================================== 特殊属性与导入 ==================================================
# __name__
# __doc__
__all__ = [
    # 分类任务指标
    "compute_Accuracy",
    "compute_Precision",
    "compute_Recall",
    "compute_F1",
    "compute_AUC_ROC",
    "compute_MCC",
    "compute_confusion_matrix",
    "compute_classification_report",
    "GET_CATEGORICAL_METRICS",

    # 回归任务指标
    "compute_MSE",
    "compute_RMSE",
    "compute_MAE",
    "compute_R2",
    "GET_REGRESSION_METRICS",

    # 聚类任务指标
    "compute_silhouette",
]
__author__ = '陆家立'
__email__ = '996153075@qq.com'
__version__ = '1.0.0'

from typing import Any, Union
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    silhouette_score,
    confusion_matrix,
    classification_report,
)

from evolution import template

# ================================================== 特殊属性与导入 ==================================================


# ================================================== 全局变量 ==================================================
RANDOM_NUMBER = 42
# ================================================== 全局变量 ==================================================


# ================================================== 函数定义 ==================================================


# ================================================== 函数定义 ==================================================
# ============================== 分类任务指标 ==============================
def compute_Accuracy(y_true, y_pred, decimal: Union[int] = 3):
    """
    计算准确率

    :param y_true: 正确标签
    :param y_pred: 预测标签
    :param decimal: 保留小位数
    :return: 准确率
    """
    acc = accuracy_score(y_true, y_pred)
    if decimal or decimal == 0:
        return round(float(acc), decimal)
    else:
        return acc


def compute_Precision(y_true, y_pred, decimal: Union[int] = 3):
    """
    计算精确率

    :param y_true: 正确标签
    :param y_pred: 预测标签
    :param decimal: 保留小位数
    :return: 精确率
    """
    precision = precision_score(y_true, y_pred)
    if decimal or decimal == 0:
        return round(float(precision), decimal)
    else:
        return precision


def compute_Recall(y_true, y_pred, decimal: Union[int] = 3):
    """
    计算召回率

    :param y_true: 正确标签
    :param y_pred: 预测标签
    :param decimal: 保留小位数
    :return: 召回率
    """
    recall = recall_score(y_true, y_pred)
    if decimal or decimal == 0:
        return round(float(recall), decimal)
    else:
        return recall


def compute_F1(y_true, y_pred, decimal: Union[int] = 3):
    """
    计算F1分数

    :param y_true: 正确标签
    :param y_pred: 预测标签
    :param decimal: 保留小位数
    :return: F1分数
    """
    f1 = f1_score(y_true, y_pred)
    if decimal or decimal == 0:
        return round(float(f1), decimal)
    else:
        return f1


def compute_AUC_ROC(y_true, y_pred, decimal: Union[int] = 3):
    """
    计算ROC曲线下的面积

    :param y_true: 正确标签
    :param y_pred: 预测标签
    :param decimal: 保留小位数
    :return: ROC曲线下的面积
    """
    roc_auc = roc_auc_score(y_true, y_pred)
    if decimal or decimal == 0:
        return round(float(roc_auc), decimal)
    else:
        return roc_auc


def compute_MCC(y_true, y_pred, decimal: Union[int] = 3):
    """
    计算Matthews相关系数

    :param y_true: 正确标签
    :param y_pred: 预测标签
    :param decimal: 保留小位数
    :return: Matthews相关系数
    """
    mcc = matthews_corrcoef(y_true, y_pred)
    if decimal or decimal == 0:
        return round(float(mcc), decimal)
    else:
        return mcc


def compute_confusion_matrix(y_true, y_pred):
    """
    计算混淆矩阵

    :param y_true: 正确标签
    :param y_pred: 预测标签
    :return: 混淆矩阵
    """
    return confusion_matrix(y_true, y_pred)


def compute_classification_report(y_true, y_pred):
    """
    计算分类报告

    :param y_true: 正确标签
    :param y_pred: 预测标签
    :return: 分类报告
    """
    return classification_report(y_true, y_pred)


def GET_CATEGORICAL_METRICS(y_true, y_pred, decimal: Union[int] = 3):
    accuracy = compute_Accuracy(y_true, y_pred, decimal=decimal)
    precision = compute_Precision(y_true, y_pred, decimal=decimal)
    recall = compute_Recall(y_true, y_pred, decimal=decimal)
    f1 = compute_F1(y_true, y_pred, decimal=decimal)
    auc = compute_AUC_ROC(y_true, y_pred, decimal=decimal)
    mcc = compute_MCC(y_true, y_pred, decimal=decimal)
    cm = compute_confusion_matrix(y_true, y_pred)
    report = compute_classification_report(y_true, y_pred)
    # return accuracy, precision, recall, f1, auc, mcc, cm, report
    return {
        '准确率': accuracy,
        '精确率': precision,
        '召回率': recall,
        'F1': f1,
        'AUC': auc,
        'MCC': mcc,
        '混淆矩阵': cm,
        '分类报告': report
    }
# ============================== 分类任务指标 ==============================


# ============================== 回归任务指标 ==============================
def compute_MSE(y_true, y_pred, decimal: Union[int] = 3):
    """
    计算均方误差

    :param y_true: 正确标签
    :param y_pred: 预测标签
    :param decimal: 保留小位数
    :return: 均方误差
    """
    mse = mean_squared_error(y_true, y_pred)
    if decimal or decimal == 0:
        return round(float(mse), decimal)
    else:
        return mse


def compute_RMSE(mse, decimal: Union[int] = 3):
    """
    计算均方根误差

    :param mse: 均方误差
    :param decimal: 保留小位数
    :return: 均方根误差
    """
    rmse = np.sqrt(mse)
    if decimal or decimal == 0:
        return round(float(rmse), decimal)
    else:
        return rmse


def compute_MAE(y_true, y_pred, decimal: Union[int] = 3):
    """
    计算平均绝对误差

    :param y_true: 正确标签
    :param y_pred: 预测标签
    :param decimal: 保留小位数
    :return: 平均绝对误差
    """
    mae = mean_absolute_error(y_true, y_pred)
    if decimal or decimal == 0:
        return round(float(mae), decimal)
    else:
        return mae


def compute_R2(y_true, y_pred, decimal: Union[int] = 3):
    """
    计算R²分数

    :param y_true: 正确标签
    :param y_pred: 预测标签
    :param decimal: 保留小位数
    :return: R²分数
    """
    r2 = r2_score(y_true, y_pred)
    if decimal or decimal == 0:
        return round(float(r2), decimal)
    else:
        return r2


def compute_PCC(y_true, y_pred, decimal: Union[int] = 3):
    """
    计算皮尔逊相关系数

    :param y_true: 正确标签
    :param y_pred: 预测标签
    :param decimal: 保留小位数
    :return: 皮尔逊相关系数
    """
    pcc = pearsonr(y_true, y_pred)[0]
    if decimal or decimal == 0:
        return round(float(pcc), decimal)
    else:
        return pcc


def compute_Spearman(y_true, y_pred, decimal: Union[int] = 3):
    """
    计算斯皮尔曼等级相关系数

    :param y_true: 正确标签
    :param y_pred: 预测标签
    :param decimal: 保留小位数
    :return: 斯皮尔曼等级相关系数
    """
    prm = spearmanr(y_true, y_pred)[0]
    if decimal or decimal == 0:
        return round(float(prm), decimal)
    else:
        return prm


def compute_KendallTau(y_true, y_pred, decimal: Union[int] = 3):
    """
    计算肯德尔一致性系数[肯德尔tau-b系数]

    :param decimal:
    :param y_true: 正确标签
    :param y_pred: 预测标签
    :param decimal: 保留小位数
    :return: 肯德尔一致性系数
    """
    kdl = kendalltau(y_true, y_pred)[0]
    if decimal or decimal == 0:
        return round(float(kdl), decimal)
    else:
        return kdl


def GET_REGRESSION_METRICS(
        y_true,
        y_pred,
        current_epoch=None,
        feature_name=None,
        model_name=None,
        decimal: Union[int] = 3,
        report_dict: dict = template.ZH_REPORT_DICT,
        report_df: dict = template.ZH_REPORT_DATA_FRAME,
):
    mse = compute_MSE(y_true, y_pred, decimal=decimal)
    rmse = compute_RMSE(mse, decimal=decimal)
    mae = compute_MAE(y_true, y_pred, decimal=decimal)
    r2 = compute_R2(y_true, y_pred, decimal=decimal)
    pcc = compute_PCC(y_true, y_pred, decimal=decimal)
    spearman = compute_Spearman(y_true, y_pred, decimal=decimal)
    kendall_tau = compute_KendallTau(y_true, y_pred, decimal=decimal)

    report_df[report_dict['训练轮次[Epoch]']] = current_epoch,
    report_df[report_dict['数据特征[Feature]']] = feature_name,
    report_df[report_dict['模型名称[Model]']] = model_name,
    report_df[report_dict['平均绝对误差[MAE]']] = mae,
    report_df[report_dict['均方根误差[RMSE]']] = rmse,
    report_df[report_dict['均方误差[MSE]']] = mse,
    report_df[report_dict['决定系数[R²]']] = r2,
    report_df[report_dict['皮尔逊相关系数[PCC]']] = pcc,
    report_df[report_dict['斯皮尔曼等级相关系数[Spearman]']] = spearman,
    report_df[report_dict['肯德尔tau-b系数[KendallTau]']] = kendall_tau,
    return report_df
# ============================== 回归任务指标 ==============================


# ============================== 聚类任务指标 ==============================
def compute_silhouette(n_clusters: int = 2, x: Any = None):
    """
    计算轮廓系数

    :param n_clusters: 集群数或质心数
    :param x: 聚类数据
    :return: 轮廓系数
    """
    kmeans = KMeans(n_clusters=n_clusters).fit(x)
    labels = kmeans.labels_
    return silhouette_score(x, labels)
# ============================== 聚类任务指标 ==============================
