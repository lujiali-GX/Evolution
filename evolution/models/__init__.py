# -*- coding: utf-8 -*-

"""
----
文件名称: __init__.py
----
\\
----
模块概述: 模型设计
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
    - evolution.models
\\
\\
----
使用示例：
----
    - models.model_metrics
    - models.dl
    - models.ml
    - models.read
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
    # 模块
    'model_metrics',
    'dl',
    'ml',
    'read',
]
__author__ = '陆家立'
__email__ = '996153075@qq.com'
__version__ = '1.0.0'

import evolution.models.model_metrics as model_metrics
import evolution.models.dl as dl
import evolution.models.ml as ml
import evolution.models.read as read
# ================================================== 特殊属性与导入 ==================================================
