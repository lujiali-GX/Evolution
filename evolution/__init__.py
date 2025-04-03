# -*- coding: utf-8 -*-

"""
----
文件名称: __init__.py
----
\\
----
模块概述: 
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
    = sys
\\
\\
----
使用示例：
----
evolution.train()
evolution.predict()
evolution.main_dl()
evolution.main_ml()
--------
\\
----    
异常处理：
----
    -
\\
----
注意事项：
----
    =
\\
"""

# ================================================== 特殊属性与导入 ==================================================
# __name__
# __doc__
__all__ = [
    # 模块
    'compute',
    'data',
    'esm',
    'feature',
    'models',
    'template',
    'data_processing',
    'dl_prediction_report',
    'ml_prediction_report',
    'version',
]
__author__ = '陆家立'
__email__ = '996153075@qq.com'
__version__ = '1.0.0'

# import evolution.compute as compute
import evolution.data as data
import evolution.esm as esm # 来源于 https://github.com/facebookresearch/esm
import evolution.feature as feature
import evolution.models as models
import evolution.template as template
import evolution.data_processing as data_processing
import evolution.dl_prediction_report as dl_prediction_report
import evolution.ml_prediction_report as ml_prediction_report
import evolution.version as version
# ================================================== 特殊属性 ==================================================

# ================================================== 全局变量 ==================================================

# ================================================== 全局变量 ==================================================
