# -*- coding: utf-8 -*-

"""
----
文件名称: ml.py
----
\\
----
模块概述: 机器学习模型
----
\\
----
作   者: ljl (996153075@qq.com)
----
\\
----
创建日期: 2025/3/29
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
    - ml.GNBReg()
    - ml.LRReg()
    - ml.LogReg()
    - ml.DTReg()
    - ml.RFReg()
    - ml.SVMReg()
    - ml.GBReg()
    - ml.XGBoostReg()
    - ml.train()
    - ml.predict()
    - ml.get_report()
    - ml.find_best_model()
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
    # 变量

    # 函数
    'train',
    'predict',
    'find_best_model',

    # 类
    'LRReg',
    'SVMReg',
    'RFReg',
    'GBReg',
    'XGBoostReg',
    'KNNReg',
    'MLPReg',
]
__author__ = '陆家立'
__email__ = '996153075@qq.com'
__version__ = '1.0.0'

from typing import Union

import xgboost
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from evolution import data
# ================================================== 特殊属性与导入 ==================================================


# ================================================== 全局变量 ==================================================

# ================================================== 全局变量 ==================================================


# ================================================== 函数定义 ==================================================
def find_best_model(x_train, x_test, y_train, y_test, models, param_grids, scoring='neg_mean_squared_error'):
    """
    查找最佳型号最佳模型

    :param x_train: 训练特征
    :param x_test: 测试特征
    :param y_train: 训练标签
    :param y_test: 测试标签
    :param models: 模型列表
    :param param_grids: 模型参数字典
    :param scoring:
    :return:
    """
    best_model = None
    best_score = float('-inf')
    best_params = {}

    for model_class, params in zip(models, param_grids[models.NAME]):
        model = model_class(x_train, x_test, y_train, y_test, show_report=False)
        grid_search = GridSearchCV(estimator=model.model, param_grid=params, cv=5, scoring=scoring)
        grid_search.fit(x_train, y_train.ravel())
        print(f"[{models.NAME}]模型最佳分数[{grid_search.best_params_}]\n最佳参数:{grid_search.best_params_}")

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = model_class
            best_params = grid_search.best_params_

    print("最佳模型: ", best_model)
    print("最佳参数: ", best_params)
    print("最佳分数: ", best_score)

    return best_model, best_params, best_score


def train( model, x_train, y_train):
    model.fit(x_train, y_train.ravel())
    return model


def predict(model, x_test):
    return model.predict(x_test)
# ================================================== 函数定义 ==================================================


# ================================================== 类定义 ==================================================
class LRReg:
    NAME = "线性回归模型"
    model = LinearRegression

    def __init__(
            self,
            x_train, x_test, y_train, y_test,
            feature_name: str = None,
            decimal: Union[int] = 3,
            show_report: bool = True,
            metrics_type='zh',
            *args, **kwargs):
        """
        线性回归模型

        :param x_train: 训练特征
        :param y_train: 训练标签
        :param x_test: 测试特征
        :param y_test: 测试标签
        :param feature_name: 特征名称
        :param metrics_type: 指标键名语言, ['zh' 中文, 'en' 英文]
        :param show_report: 是否获取报告
        :param args: 参数
        :param kwargs: 参数关键值
        :return:
        """
        model = self.model(*args, **kwargs)
        self.y_true = y_test
        self.feature_name = feature_name
        self.decimal = decimal if decimal is not None else 3
        self.show_report = show_report
        self.metrics_type = metrics_type

        self.model = train(model=model, x_train=x_train, y_train=y_train)
        self.y_pred = predict(model=self.model, x_test=x_test)
        self.report = data.get_report(
            y_true=self.y_true, y_pred=self.y_pred,
            data_name=self.NAME,
            feature_name=self.feature_name,
            model_name=self.NAME,
            decimal=self.decimal,
            metrics_type=self.metrics_type,
            show_report=self.show_report,
        )
    def fix(self, x_train, y_train):
        self.model = train(model=self.model, x_train=x_train, y_train=y_train)
    def predict(self, x_test):
        self.y_pred = predict(model=self.model, x_test=x_test)
        return self.y_pred
    def get_report(self, decimal: int = 3, metrics_type: str = 'zh', show_report: bool = False):
        self.report = data.get_report(
            y_true=self.y_true, y_pred=self.y_pred,
            data_name=self.NAME,
            feature_name=self.feature_name,
            model_name=self.NAME,
            decimal=decimal,
            metrics_type=metrics_type,
            show_report=show_report,
        )
        return self.report


class DTReg:
    NAME = "决策树回归模型"
    model = DecisionTreeRegressor

    def __init__(
            self,
            x_train, x_test, y_train, y_test,
            feature_name: str = None,
            decimal: Union[int] = 3,
            show_report: bool = True,
            metrics_type='zh',
            *args, **kwargs):
        """
        决策树回归模型

        :param x_train: 训练特征
        :param y_train: 训练标签
        :param x_test: 测试特征
        :param y_test: 测试标签
        :param feature_name: 特征名称
        :param metrics_type: 指标键名语言, ['zh' 中文, 'en' 英文]
        :param show_report: 是否获取报告
        :param args: 参数
        :param kwargs: 参数关键值
        :return:
        """
        model = self.model(*args, **kwargs)
        self.y_true = y_test
        self.feature_name = feature_name
        self.decimal = decimal if decimal is not None else 3
        self.show_report = show_report
        self.metrics_type = metrics_type

        self.model = train(model=model, x_train=x_train, y_train=y_train)
        self.y_pred = predict(model=self.model, x_test=x_test)
        self.report = data.get_report(
            y_true=self.y_true, y_pred=self.y_pred,
            data_name=self.NAME,
            feature_name=self.feature_name,
            model_name=self.NAME,
            decimal=self.decimal,
            metrics_type=self.metrics_type,
            show_report=self.show_report,
        )

    def fix(self, x_train, y_train):
        self.model = train(model=self.model, x_train=x_train, y_train=y_train)
    def predict(self, x_test):
        self.y_pred = predict(model=self.model, x_test=x_test)
        return self.y_pred
    def get_report(self, decimal: int = 3, metrics_type: str = 'zh', show_report: bool = False):
        self.report = data.get_report(
            y_true=self.y_true, y_pred=self.y_pred,
            data_name=self.NAME,
            feature_name=self.feature_name,
            model_name=self.NAME,
            decimal=decimal,
            metrics_type=metrics_type,
            show_report=show_report,
        )
        return self.report


class RFReg:
    NAME = "随机森林回归模型"
    model = RandomForestRegressor

    def __init__(
            self,
            x_train, x_test, y_train, y_test,
            feature_name: str = None,
            decimal: Union[int] = 3,
            show_report: bool = True,
            metrics_type='zh',
            *args, **kwargs):
        """
        随机森林回归模型

        :param x_train: 训练特征
        :param y_train: 训练标签
        :param x_test: 测试特征
        :param y_test: 测试标签
        :param feature_name: 特征名称
        :param metrics_type: 指标键名语言, ['zh' 中文, 'en' 英文]
        :param show_report: 是否获取报告
        :param args: 参数
        :param kwargs: 参数关键值
        :return:
        """
        model = self.model(*args, **kwargs)
        self.y_true = y_test
        self.feature_name = feature_name
        self.decimal = decimal if decimal is not None else 3
        self.show_report = show_report
        self.metrics_type = metrics_type

        self.model = train(model=model, x_train=x_train, y_train=y_train)
        self.y_pred = predict(model=self.model, x_test=x_test)
        self.report = data.get_report(
            y_true=self.y_true, y_pred=self.y_pred,
            data_name=self.NAME,
            feature_name=self.feature_name,
            model_name=self.NAME,
            decimal=self.decimal,
            metrics_type=self.metrics_type,
            show_report=self.show_report,
        )

    def fix(self, x_train, y_train):
        self.model = train(model=self.model, x_train=x_train, y_train=y_train)
    def predict(self, x_test):
        self.y_pred = predict(model=self.model, x_test=x_test)
        return self.y_pred
    def get_report(self, decimal: int = 3, metrics_type: str = 'zh', show_report: bool = False):
        self.report = data.get_report(
            y_true=self.y_true, y_pred=self.y_pred,
            data_name=self.NAME,
            feature_name=self.feature_name,
            model_name=self.NAME,
            decimal=decimal,
            metrics_type=metrics_type,
            show_report=show_report,
        )
        return self.report


class SVMReg:
    NAME = "支持向量机回归模型"
    model = SVR

    def __init__(
            self,
            x_train, x_test, y_train, y_test,
            feature_name: str = None,
            decimal: Union[int] = 3,
            show_report: bool = True,
            metrics_type='zh',
            *args, **kwargs):
        """
        支持向量机回归模型

        :param x_train: 训练特征
        :param y_train: 训练标签
        :param x_test: 测试特征
        :param y_test: 测试标签
        :param feature_name: 特征名称
        :param metrics_type: 指标键名语言, ['zh' 中文, 'en' 英文]
        :param show_report: 是否获取报告
        :param args: 参数
        :param kwargs: 参数关键值
        :return:
        """
        model = self.model(*args, **kwargs)
        self.y_true = y_test
        self.feature_name = feature_name
        self.decimal = decimal if decimal is not None else 3
        self.show_report = show_report
        self.metrics_type = metrics_type

        self.model = train(model=model, x_train=x_train, y_train=y_train)
        self.y_pred = predict(model=self.model, x_test=x_test)
        self.report = data.get_report(
            y_true=self.y_true, y_pred=self.y_pred,
            data_name=self.NAME,
            feature_name=self.feature_name,
            model_name=self.NAME,
            decimal=self.decimal,
            metrics_type=self.metrics_type,
            show_report=self.show_report,
        )

    def fix(self, x_train, y_train):
        self.model = train(model=self.model, x_train=x_train, y_train=y_train)
    def predict(self, x_test):
        self.y_pred = predict(model=self.model, x_test=x_test)
        return self.y_pred
    def get_report(self, decimal: int = 3, metrics_type: str = 'zh', show_report: bool = False):
        self.report = data.get_report(
            y_true=self.y_true, y_pred=self.y_pred,
            data_name=self.NAME,
            feature_name=self.feature_name,
            model_name=self.NAME,
            decimal=decimal,
            metrics_type=metrics_type,
            show_report=show_report,
        )
        return self.report


class GBReg:
    NAME = "梯度提升回归模型"
    model = GradientBoostingRegressor

    def __init__(
            self,
            x_train, x_test, y_train, y_test,
            feature_name: str = None,
            decimal: Union[int] = 3,
            show_report: bool = True,
            metrics_type='zh',
            *args, **kwargs):
        """
        支持向量机回归模型

        :param x_train: 训练特征
        :param y_train: 训练标签
        :param x_test: 测试特征
        :param y_test: 测试标签
        :param feature_name: 特征名称
        :param metrics_type: 指标键名语言, ['zh' 中文, 'en' 英文]
        :param show_report: 是否获取报告
        :param args: 参数
        :param kwargs: 参数关键值
        :return:
        """
        model = self.model(*args, **kwargs)
        self.y_true = y_test
        self.feature_name = feature_name
        self.decimal = decimal if decimal is not None else 3
        self.show_report = show_report
        self.metrics_type = metrics_type

        self.model = train(model=model, x_train=x_train, y_train=y_train)
        self.y_pred = predict(model=self.model, x_test=x_test)
        self.report = data.get_report(
            y_true=self.y_true, y_pred=self.y_pred,
            data_name=self.NAME,
            feature_name=self.feature_name,
            model_name=self.NAME,
            decimal=self.decimal,
            metrics_type=self.metrics_type,
            show_report=self.show_report,
        )

    def fix(self, x_train, y_train):
        self.model = train(model=self.model, x_train=x_train, y_train=y_train)
    def predict(self, x_test):
        self.y_pred = predict(model=self.model, x_test=x_test)
        return self.y_pred
    def get_report(self, decimal: int = 3, metrics_type: str = 'zh', show_report: bool = False):
        self.report = data.get_report(
            y_true=self.y_true, y_pred=self.y_pred,
            data_name=self.NAME,
            feature_name=self.feature_name,
            model_name=self.NAME,
            decimal=decimal,
            metrics_type=metrics_type,
            show_report=show_report,
        )
        return self.report


class XGBoostReg:
    NAME = "极端梯度提升回归模型"
    model = xgboost.XGBRegressor

    def __init__(
            self,
            x_train, x_test, y_train, y_test,
            feature_name: str = None,
            decimal: Union[int] = 3,
            show_report: bool = True,
            metrics_type='zh',
            *args, **kwargs):
        """
        极端梯度提升回归模型

        :param x_train: 训练特征
        :param y_train: 训练标签
        :param x_test: 测试特征
        :param y_test: 测试标签
        :param feature_name: 特征名称
        :param metrics_type: 指标键名语言, ['zh' 中文, 'en' 英文]
        :param show_report: 是否获取报告
        :param args: 参数
        :param kwargs: 参数关键值
        :return:
        """
        model = self.model(*args, **kwargs)
        self.y_true = y_test
        self.feature_name = feature_name
        self.decimal = decimal if decimal is not None else 3
        self.show_report = show_report
        self.metrics_type = metrics_type

        self.model = train(model=model, x_train=x_train, y_train=y_train)
        self.y_pred = predict(model=self.model, x_test=x_test)
        self.report = data.get_report(
            y_true=self.y_true, y_pred=self.y_pred,
            data_name=self.NAME,
            feature_name=self.feature_name,
            model_name=self.NAME,
            decimal=self.decimal,
            metrics_type=self.metrics_type,
            show_report=self.show_report,
        )

    def fix(self, x_train, y_train):
        self.model = train(model=self.model, x_train=x_train, y_train=y_train)
    def predict(self, x_test):
        self.y_pred = predict(model=self.model, x_test=x_test)
        return self.y_pred
    def get_report(self, decimal: int = 3, metrics_type: str = 'zh', show_report: bool = False):
        self.report = data.get_report(
            y_true=self.y_true, y_pred=self.y_pred,
            data_name=self.NAME,
            feature_name=self.feature_name,
            model_name=self.NAME,
            decimal=decimal,
            metrics_type=metrics_type,
            show_report=show_report,
        )
        return self.report


class KNNReg:
    NAME = "K近邻回归"
    model = KNeighborsRegressor

    def __init__(
            self,
            x_train, x_test, y_train, y_test,
            feature_name: str = None,
            decimal: Union[int] = 3,
            show_report: bool = True,
            metrics_type='zh',
            n_neighbors: int = 5,
            *args, **kwargs):
        """
        支持向量机回归模型

        :param x_train: 训练特征
        :param y_train: 训练标签
        :param x_test: 测试特征
        :param y_test: 测试标签
        :param feature_name: 特征名称
        :param metrics_type: 指标键名语言, ['zh' 中文, 'en' 英文]
        :param show_report: 是否获取报告
        :param args: 参数
        :param kwargs: 参数关键值
        :return:
        """
        model = self.model(n_neighbors=n_neighbors, *args, **kwargs)
        self.y_true = y_test
        self.feature_name = feature_name
        self.decimal = decimal if decimal is not None else 3
        self.show_report = show_report
        self.metrics_type = metrics_type

        self.model = train(model=model, x_train=x_train, y_train=y_train)
        self.y_pred = predict(model=self.model, x_test=x_test)
        self.report = data.get_report(
            y_true=self.y_true, y_pred=self.y_pred,
            data_name=self.NAME,
            feature_name=self.feature_name,
            model_name=self.NAME,
            decimal=self.decimal,
            metrics_type=self.metrics_type,
            show_report=self.show_report,
        )

    def fix(self, x_train, y_train):
        self.model = train(model=self.model, x_train=x_train, y_train=y_train)
    def predict(self, x_test):
        self.y_pred = predict(model=self.model, x_test=x_test)
        return self.y_pred
    def get_report(self, decimal: int = 3, metrics_type: str = 'zh', show_report: bool = False):
        self.report = data.get_report(
            y_true=self.y_true, y_pred=self.y_pred,
            data_name=self.NAME,
            feature_name=self.feature_name,
            model_name=self.NAME,
            decimal=decimal,
            metrics_type=metrics_type,
            show_report=show_report,
        )
        return self.report


class MLPReg:
    NAME = "神经网络回归模型"
    model = MLPRegressor

    def __init__(
            self,
            x_train, x_test, y_train, y_test,
            feature_name: str = None,
            decimal: Union[int] = 3,
            show_report: bool = True,
            metrics_type='zh',
            *args, **kwargs):
        """
        神经网络回归模型

        :param x_train: 训练特征
        :param y_train: 训练标签
        :param x_test: 测试特征
        :param y_test: 测试标签
        :param feature_name: 特征名称
        :param metrics_type: 指标键名语言, ['zh' 中文, 'en' 英文]
        :param show_report: 是否获取报告
        :param args: 参数
        :param kwargs: 参数关键值
        :return:
        """
        model = self.model(*args, **kwargs)
        self.y_true = y_test
        self.feature_name = feature_name
        self.decimal = decimal if decimal is not None else 3
        self.show_report = show_report
        self.metrics_type = metrics_type

        self.model = train(model=model, x_train=x_train, y_train=y_train)
        self.y_pred = predict(model=self.model, x_test=x_test)
        self.report = data.get_report(
            y_true=self.y_true, y_pred=self.y_pred,
            data_name=self.NAME,
            feature_name=self.feature_name,
            model_name=self.NAME,
            decimal=self.decimal,
            metrics_type=self.metrics_type,
            show_report=self.show_report,
        )

    def fix(self, x_train, y_train):
        self.model = train(model=self.model, x_train=x_train, y_train=y_train)
    def predict(self, x_test):
        self.y_pred = predict(model=self.model, x_test=x_test)
        return self.y_pred
    def get_report(self, decimal: int = 3, metrics_type: str = 'zh', show_report: bool = False):
        self.report = data.get_report(
            y_true=self.y_true, y_pred=self.y_pred,
            data_name=self.NAME,
            feature_name=self.feature_name,
            model_name=self.NAME,
            decimal=decimal,
            metrics_type=metrics_type,
            show_report=show_report,
        )
        return self.report


ML_MODEL_DICT = {
    "LR": LRReg,
    "DT": DTReg,
    "RF": RFReg,
    "SVM": SVMReg,
    "GB": GBReg,
    "XGB": XGBoostReg,
    "KNN": KNNReg,
    "MLP": MLPReg,

}
"""机器学习模型字典"""
