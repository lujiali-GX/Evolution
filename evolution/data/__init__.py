# -*- coding: utf-8 -*-

"""
----
文件名称: __init__.py
----
\\
----
模块概述: 数据处理模块
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
    - json
    - os
    - typing.Any as Any
    - numpy as np
    - pandas as pd
    - torch
    - torch.nn as nn
    - torch.utils.data.Dataset as Dataset
\\
----
使用示例：
----
    - data.load_dataset(data_path=os.path.join(template.TEMP_DIR, 'PG16_P.pt'), device=torch.device('cpu'))
    - data.divide_data_set(data_path=os.path.join(template.TEMP_DIR, 'PG16_P.pt'), ratio=(0.70, 0.15, 0.15), device=torch.device('cpu'))
    - data.add_noise(np.array([1, 2, 3]), 0.01)
    - data.save_json_data({"test": "Hello World!"}, path="../exclude/test.json")
    - data.save_npy_data([0, 1, 2, 3, 4], path="../exclude/test.npy")
    - data.save_df_data([0, 1, 2, 3, 4], path="../exclude/test.csv")
    - data.save_torch_data([0, 1, 2, 3, 4], path="../exclude/test.pt")
    - data.get_json_data(path='../exclude/test.json', out_data_type=outDataType)
    - data.data.get_npy_data(path='../exclude/test.npy', out_data_function=np.array)
    - data.get_df_data(path='../exclude/test.csv', out_data_function=pd.DataFrame)
    - data.get_torch_data(path='../exclude/test.pt', out_data_function=torch.tensor)
\\
\\
----    
异常处理：
----
    - ValueError
    - TypeError
    - Exception

    - json.JSONDecodeError
    - np.exceptions.AxisError
    - pd.errors.ParserError
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
    'get_report',  # 获取数据报告
    'show_metrics',  # 打印数据报告
    'load_dataset',  # 加载数据集
    'divide_data_set',  # 划分数据集
    'add_noise',  # 数据降噪
    'get_json_data',  # 获取json数据(字典)
    'get_npy_data',  # 获取npy数据(数组)
    'get_torch_data',  # 获取torch数据(张量)
    'get_df_data',  # 获取df数据(DataFrame)
    'save_json_data',  # 保存json数据
    'save_npy_data',  # 保存npy数据
    'save_df_data',  # 保存df数据
    'save_torch_data',  # 保存torch数据
]
__author__ = '陆家立'
__email__ = '996153075@qq.com'
__version__ = '1.0.0'

import json
import os
from typing import Any, Union

import numpy as np
import pandas as pd
import torch
from pandas.errors import ParserError
from torch import nn
from torch.utils.data import Dataset, random_split

import evolution.data.preconditioning as preconditioning
from evolution import template
from evolution.models.model_metrics import GET_REGRESSION_METRICS


# ================================================== 特殊属性与导入 ==================================================


# ================================================== 全局变量 ==================================================

# ================================================== 全局变量 ==================================================


# ================================================== 函数定义 ==================================================
def get_report(
        y_true, y_pred,
        current_epoch: Union[int] = None,
        data_name: Union[str] = None,
        feature_name: Union[str] = None,
        model_name: Union[str] = None,
        decimal: Union[int] = None,
        metrics_type: str = 'zh',
        show_report: Union[bool] = None
):
    """
    获取报告

    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param current_epoch: 当前轮次
    :param data_name: 数据名称
    :param feature_name: 特征名称
    :param model_name: 模型名称
    :param decimal: 保留小位数
    :param metrics_type: 指标键名语言, ['zh' 中文, 'en' 英文]
    :param show_report: 是否显示
    :return:
    """
    if metrics_type in ['zh', 'chinese']:
        report_dict = template.ZH_REPORT_DICT
        report_df = template.ZH_REPORT_DATA_FRAME
    elif metrics_type in ['en', 'english']:
        report_dict = template.EN_REPORT_DICT
        report_df = template.EN_REPORT_DATA_FRAME
    else:
        report_dict = template.ZH_REPORT_DICT
        report_df = template.ZH_REPORT_DATA_FRAME
    report_df = GET_REGRESSION_METRICS(
        y_true=y_pred,
        y_pred=y_true,
        current_epoch=current_epoch,
        feature_name=feature_name,
        model_name=model_name,
        decimal=decimal,
        report_dict=report_dict,
        report_df=report_df,
    )
    if show_report:
        show_metrics(metrics=report_df, data_name=data_name, feature_name=feature_name)
    return report_df


def show_metrics(
        metrics: pd.DataFrame,
        data_name: str = "模型",
        feature_name: str = "特征名称",
):
    """
    展示报告

    :param metrics: 指标数据
    :param data_name: 数据名称
    :param feature_name: 数据特征
    :raises TypeError: 报告数据类型错误
    :return:
    """
    if not isinstance(metrics, pd.DataFrame):
        raise TypeError(f"报告数据类型 {type(metrics)} 错误, 必须是DataFrame类型")
    print(f"[{data_name}] 数据 [{feature_name}] 特征评估报告:")
    for key, value in metrics.items():
        print(f"\n\t{key}: {value}\n")
    return True


def load_dataset(
        data_path: str = None,
        device: torch.device = template.SuperParameters.device
):
    """
    加载数据集

    :example:
    >>> result = load_dataset(data_path=os.path.join(template.TEMP_DIR, 'PG16_P.pt'), device=torch.device('cpu'))
    >>> isinstance(result, list)
    True

    :param data_path: 数据集路径
    :param device: 加载设备
    :return:
    """
    try:
        return torch.load(data_path, map_location=device, weights_only=False)
    except torch.cuda.CudaError:
        print("加速计算失败, GPU 显存不足，自动切换到CPU")
        return torch.load(data_path, map_location=torch.device('cpu'))
    except Exception as e:
        raise Exception(f"其他错误: {e}")


def divide_data_set(
        data: Any = None,
        data_path: str = None,
        ratio: tuple = (0.70, 0.15, 0.15),
        device: torch.device = template.SuperParameters.device
) :
    """
    数据集划分

    :example:
    >>> from evolution import template
    >>> result = divide_data_set(
    ...     data_path=os.path.join(template.TEMP_DIR, 'PG16_P.pt'),
    ...     ratio=(0.70, 0.15, 0.15),
    ...     device=torch.device('cpu'))
    >>> isinstance(result, tuple)
    True

    :param data: 数据集路径
    :param data_path: 数据集路径
    :param ratio: 划分比例
    :param device: 设备
    :return: 训练集, 验证集, 测试集
    """
    if data_path is not None:
        load_data = load_dataset(data_path, device)
    elif data is not None:
        load_data = data
    else:
        raise Exception("数据集路径或数据集为空")

    # 计算数据集划分的长度
    train_size = int(ratio[0] * len(load_data))
    val_size = int(ratio[1] * len(load_data))
    test_size = len(load_data) - train_size - val_size

    try:
        train_dataset, val_dataset, test_dataset = random_split(
            load_data, [train_size, val_size, test_size])
    except Exception as e:
        raise Exception(f"数据集划分错误: {e}")

    return train_dataset, val_dataset, test_dataset


def add_noise(array: np.ndarray, noise_level=0.01) -> np.ndarray:
    """
    数据降噪

    :examples:
    >>> result = add_noise(np.array([0, 1, 2, 3, 4]), 0.01)
    >>> isinstance(result, np.ndarray)
    True

    :param array: 数组
    :param noise_level: 噪声等级
    :return: 降噪后的数组
    :raises TypeError: array 必须是 numpy 数组
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("array 必须是 numpy 数组")
    noise = np.random.normal(0, noise_level, array.shape)
    return array + noise


def get_json_data(path: str = '../exclude/test.json', out_data_type: type = dict):
    """
    获取 json 数据

    :example: 排除
    >>> outDataType = dict
    >>> result = get_json_data(path='../exclude/test.json', out_data_type=outDataType)
    >>> isinstance(result, outDataType)
    True

    :param path: json 文件路径
    :param out_data_type: 转换格式
    :return: json 数据
    :raise FileNotFoundError: 文件错误
    :raise ValueError: 数据错误
    :raise TypeError: 格式错误
    :raise Exception: 其他错误
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件{path}不存在")
    if not path.endswith('.json'):
        raise ValueError("文件格式错误")
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        try:
            return out_data_type(data)
        except ValueError as e:
            raise ValueError("数据格式错误: {}".format(e))
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError("格式错误: {}".format(e.msg), e.doc, e.pos)
    except TypeError as e:
        raise TypeError("格式错误: {}".format(e))
    except Exception as e:
        raise Exception("其他错误: {}".format(e))


def get_npy_data(path: str , out_data_function: Any = np.array):
    """
    获取 npy 数据

    :example: 排除
    >>> result = get_npy_data(path='../exclude/test.npy', out_data_function=np.array)
    >>> isinstance(result, np.ndarray)
    True

    :param path: npy 文件路径
    :param out_data_function: 输出 npy 数据的方法
    :return: npy 数据
    :raise FileNotFoundError: 文件不存在
    :raise ValueError: 数据错误
    :raise TypeError: 格式错误
    :raise Exception: 其他错误
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件{path}不存在")
    try:
        data = np.load(path)
        try:
            return out_data_function(data)
        except ValueError as e:
            raise ValueError("数据错误: {}".format(e))
    except np.exceptions.AxisError as e:
        raise ValueError("格式错误: {}".format(e))
    except TypeError as e:
        raise TypeError("格式错误: {}".format(e))
    except Exception as e:
        raise Exception("其他错误: {}".format(e))


def get_df_data(path: str , out_data_function: Any = pd.DataFrame):
    """
    获取 DataFrame 数据

    :example: 排除
    >>> result = get_df_data(path='../exclude/test.csv', out_data_function=pd.DataFrame)
    >>> isinstance(result, pd.DataFrame)
    True

    :param path: DataFrame 数据文件路径
    :param out_data_function: 输出 DataFrame 数据的方法
    :return: DataFrame 数据
    :raise FileNotFoundError: 文件不存在
    :raise ValueError: 数据错误
    :raise TypeError: 格式错误
    :raise Exception: 其他错误
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件{path}不存在")
    if os.path.splitext(path)[1] not in ['.csv', '.xlsx', '.txt']:
        raise ValueError("文件格式错误")
    try:
        if path.endswith('.csv'):
            data = pd.read_csv(path)
        elif path.endswith('.xlsx'):
            data = pd.read_excel(path)
        elif path.endswith('.txt'):
            data = pd.read_table(path)
        else:
            raise TypeError("文件格式错误")
        try:
            return out_data_function(data)
        except ValueError as e:
            raise ValueError("数据错误: {}".format(e))
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError("格式错误: {}".format(e))
    except Exception as e:
        raise Exception("其他错误: {}".format(e))


def get_torch_data(path: str , out_data_function: Any = torch.tensor):
    """
    获取 torch 数据

    :example: 排除
    >>> result = get_torch_data(path='../exclude/test.pt', out_data_function=torch.tensor)
    >>> isinstance(result, torch.Tensor)
    True

    :param path: torch 数据文件路径
    :param out_data_function: 输出 torch 数据的方法
    :return: torch 数据
    :raise FileNotFoundError: 文件不存在
    :raise ImportError: 导入错误
    :raise ValueError: 数据错误
    :raise TypeError: 格式错误
    :raise Exception: 其他错误
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件{path}不存在")
    if os.path.splitext(path)[1] not in ['.pt', '.pth', '.onnx']:
        raise ValueError("文件格式错误")
    try:
        import onnx
        from onnx2torch import convert

        data = None
        if path.endswith(('.pt', '.pth')):
            data = torch.load(path)
        elif path.endswith('.onnx'):
            onnx_model = onnx.load(path)
            # 将ONNX模型转换为PyTorch模型
            data = convert(onnx_model)

        return out_data_function(data)
    except ValueError as e:
        raise ValueError("数据错误: {}".format(e))
    except TypeError as e:
        raise TypeError("格式错误: {}".format(e))
    except ImportError as e:
        raise ImportError("导入错误, 请安装 onnx 和 onnx2torch, 报错信息: {}".format(e))
    except Exception as e:
        raise Exception("其他错误: {}".format(e))


def gen_report(df: pd.DataFrame, save_path: str):
    """生成报告

    :example:
    >>> result = gen_report(df=template.ZH_REPORT_DATA_FRAME, save_path='../exclude/test.xlsx')
    >>> True

    :param df:
    :param save_path:
    :return:
    """
    assert isinstance(df, pd.DataFrame), f"类型错误[{type(df)}]: df 必须是 pandas.DataFrame"
    if save_path.endswith((".xlsx", ".xls")):
        df.to_excel(save_path, index=False)
    elif save_path.endswith((".csv", ".txt")):
        df.to_csv(save_path, index=False)
    elif df.endswith((".json", ".js")):
        df.to_json(save_path, orient='records', lines=True)
    return True


def save_report(report, save_path, language="zh"):
    """添加报告

    :example:
    >>> result = save_report(report=template.ZH_REPORT_DATA_FRAME, save_path='../exclude/test.xlsx', language='en')
    >>> True

    :param report:
    :param save_path:
    :param language:
    :return:
    :raise ParserError: 格式错误
    :raise TypeError: 类型错误
    :raise ValueError: 数据错误
    :raise PermissionError: 权限错误
    :raise 内存错误: 内存错误
    """
    if not isinstance(report, (dict, pd.DataFrame)):
        raise ValueError("report 必须是 dict 或 pd.DataFrame")
    if isinstance(report, dict):
        report = pd.DataFrame(data=report, columns=list(report.keys()), index=[0])
    if language == "zh":
        report = report.rename(columns=template.ZH_REPORT_DICT)
    elif language == "en":
        report = report.rename(columns=template.EN_REPORT_DICT)
    else:
        raise ValueError("language 必须是 zh 或 en")
    try:
        if template.is_exists_file(save_path):
            df = pd.read_excel(save_path)
            if language == "zh":
                df = df.rename(columns=template.ZH_REPORT_DICT)
            elif language == "en":
                df = df.rename(columns=template.EN_REPORT_DICT)
            df = pd.concat([df, report], axis=0, ignore_index=True)
        else:
            df = report
        return gen_report(df=df, save_path=save_path)
    except ParserError as e: raise ParserError(f"格式错误: {save_path} 文件格式不正确, 文件读取失败")
    except TypeError as e: raise TypeError(f"类型错误: {save_path} 文件格式不正确, 文件读取失败")
    except ValueError as e: raise ValueError(f"数据错误: {save_path} 数据列名不一致, 添加数据失败")
    except PermissionError as e: raise PermissionError(f"权限错误: {save_path} 权限不足, 文件读取失败")
    except MemoryError as e: raise MemoryError(f"内存错误: {save_path} 内存不足, 文件读取失败")


def save_json_data(data, path: str):
    """
    保存 json 数据文件

    :example:
    >>> result = save_json_data({"test": "Hello World!"}, path="../exclude/test.json")
    >>> result
    True

    :param data: 数据
    :param path: 输出文件
    :return:
    :raise TypeError: 格式错误
    :raise json.JSONDecodeError: 数据错误
    :raise Exception: 格式错误
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    if not path.endswith(('.json', '.jsonl', 'geojson', 'custom', '.txt')):
        raise TypeError("格式错误: {}".format(os.path.splitext(path)[1]))
    try:
        if isinstance(data, pd.DataFrame):
            for key, items in data.items():
                data[key] = items.values.tolist()
        with open(path, 'w') as f:
            f: Any = f
            json.dump(data, f, )
    except json.JSONDecodeError  as e:
        raise json.JSONDecodeError("数据错误: {}".format(e), f.read(), 0)
    except Exception as e:
        raise Exception("其他错误: {}".format(e))
    return True


def save_npy_data(data, path: str):
    """
    保存 npy 数据文件

    :example:
    >>> result = save_npy_data([0, 1, 2, 3, 4], path="../exclude/test.npy")
    >>> result
    True

    :param data: 数据
    :param path: 输出文件
    :return:
    :raise TypeError: 格式错误
    :raise np.exceptions.AxisError: 数据错误
    :raise Exception: 其他错误
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    if not path.endswith(('.npy', '.npyz', '.csv', '.txt')):
        raise TypeError("格式错误: {}".format(os.path.splitext(path)[1]))
    try:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        np.save(path, data)
    except np.exceptions.AxisError as e:
        raise print("数据错误: {}".format(e))
    except Exception as e:
        raise Exception("其他错误: {}".format(e))
    return True


def save_df_data(data, path: str):
    """
    保存 DataFrame 数据文件

    :example:
    >>> result = save_df_data([0, 1, 2, 3, 4], path="../exclude/test.csv")
    >>> result
    True

    :param data: 数据
    :param path: 输出文件
    :return:
    :raise TypeError: 格式错误
    :raise np.exceptions.AxisError: 数据错误
    :raise Exception: 其他错误
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    if not path.endswith(
            ('.csv', '.xls', '.xlsx', '.json', '.html', '.db', '.parquet', '.txt')
    ):
        raise TypeError("格式错误: {}".format(os.path.splitext(path)[1]))
    try:
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        if path.endswith(('.csv', '.txt')):
            data.to_csv(path, index=False)
        elif path.endswith(('.xls', '.xlsx')):
            with pd.ExcelWriter('data.xlsx') as writer:
                data.to_excel(writer, sheet_name='Sheet1', index=False)
        elif path.endswith('.json'):
            data.to_json(path, orient='records')
        elif path.endswith('.html'):
            data.to_html(path, index=False)
        elif path.endswith('.db'):
            data.to_sql(path, index=False)
        elif path.endswith('.parquet'):
            data.to_parquet(path, index=False)
        else:
            raise TypeError("格式错误: {}".format(os.path.splitext(path)[1]))
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError("数据错误: {}".format(e))
    except Exception as e:
        raise Exception("其他错误: {}".format(e))
    return True

def save_torch_data(data, path: str):
    """
    保存 torch 数据文件

    :example:
    >>> result = save_torch_data([0, 1, 2, 3, 4], path="../exclude/test.pt")
    >>> result
    True

    :param data: 数据
    :param path: 输出文件
    :return:
    :raise TypeError: 格式错误
    :raise np.exceptions.AxisError: 数据错误
    :raise Exception: 其他错误
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    if not path.endswith(('.pt', '.pth', '.onnx')):
        raise TypeError("格式错误: {}".format(os.path.splitext(path)[1]))
    try:
        if not isinstance(data, (torch.Tensor, list)):
            data = torch.tensor(data)
        if path.endswith(('.pt', '.pth')):
            torch.save(data, path)
        elif path.endswith('.onnx'):
            # 定义一个简单的模型
            class SimpleModel(nn.Module):
                def __init__(self, input_features, out_features: int = 1):
                    super(SimpleModel, self).__init__()
                    self.input_features = input_features
                    self.out_features = out_features
                    # 这里定义了 nn.Linear 层，会自动管理权重和偏置
                    self.fc = nn.Linear(
                        in_features=self.input_features,
                        out_features=self.out_features,
                        bias=True,
                    )

                def forward(self, x):
                    # 将输入张量转换为 float32 类型
                    x = x.to(torch.float32)
                    # 只传入输入张量 x
                    return self.fc(x)
            # 保证 args 参数和 forward 方法匹配
            torch.onnx.export(model=SimpleModel(data.shape[0], 1), args=(data, ), f=path)
        else:
            raise TypeError("格式错误: {}".format(os.path.splitext(path)[1]))
    except np.exceptions.AxisError as e:
        raise print("数据错误: {}".format(e))
    except Exception as e:
        raise Exception("其他错误: {}".format(e))
    return True
# ================================================== 函数定义 ==================================================
