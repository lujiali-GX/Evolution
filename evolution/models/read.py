# -*- coding: utf-8 -*-

"""
----
文件名称: read.py
----
\\
----
模块概述: 读取预训练模型
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
    - torch
    - transformers.AutoTokenizer as AutoTokenizer
\\
\\
----
使用示例：
----
    - read.get_prottrans_model()
    - read.get_esm2_tokenizer()
    - read.get_esm_model()
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
    'get_prottrans_model',
    'get_esm2_tokenizer',
    'get_esm_model',
]
__author__ = '陆家立'
__email__ = '996153075@qq.com'
__version__ = '1.0.0'

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from transformers import AutoTokenizer

from evolution import esm
from evolution import template
# ================================================== 特殊属性与导入 ==================================================


# ================================================== 全局变量 ==================================================

# ================================================== 全局变量 ==================================================


# ================================================== 函数定义 ==================================================
def get_prottrans_model(
        prottrans_dir: str = template.PROTTRANS_DIR,
):
    """
    获取 prottrans 预训练模型

    :example:
    >>> result = get_prottrans_model(template.PROTTRANS_DIR)
    >>> isinstance(result, tuple)
    True

    :param prottrans_dir: prottrans 模型目录
    :return: prottrans 预训练模型和分词器
    :raise FileNotFoundError: 文件错误
    :raise ImportError: 导入错误
    :raise Exception: 读取错误
    """
    if not os.path.exists(prottrans_dir):
        raise FileNotFoundError(f"文件错误: ['{prottrans_dir}']不存在")
    from transformers import T5Tokenizer, T5EncoderModel
    try:
        prottrans_tokenizer = T5Tokenizer.from_pretrained(prottrans_dir, legacy=False)
        prottrans_model = T5EncoderModel.from_pretrained(prottrans_dir)
        prottrans_model.to(template.DEVICE)
        if torch.cuda.device_count() > 1:
            prottrans_model = torch.nn.DataParallel(prottrans_model)
        prottrans_model.eval()
    except ImportError as e:
        raise print(f"导入错误: 请安装 transformers\n{e}")
    except Exception as e:
        raise Exception(f"读取错误: prottrans 模型加载错误\n{e}")
    return prottrans_model, prottrans_tokenizer


def get_esm2_tokenizer():
    if not os.path.exists(template.ESM2_MODEL): raise FileNotFoundError(f"['{template.ESM2_MODEL}']不存在")
    return AutoTokenizer.from_pretrained(template.ESM2_DIR)


def get_esm_model():
    esm2_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm2_model = esm2_model.to(template.DEVICE)
    esm2_model.eval()
    return esm2_model, alphabet
# ================================================== 函数定义 ==================================================
