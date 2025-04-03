# -*- coding: utf-8 -*-

"""
----
文件名称: __init__.py
----
\\
----
模块概述: 计算数据特征
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
    - os
    - numpy as np
    - pandas as pd
    - torch
    - torch_geometric.data.data.Data as Data
    - tqdm
\\
\\
----
使用示例：
----
    - compute.compute_features -- 计算数据特征 -- 详情请看方法文档
    - compute.compute_Data -- 计算单个数据特征 -- 详情请看方法文档
    - compute.compute_torch_features -- 计算 torch 数据特征 -- 详情请看方法文档
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
    'compute_features',
    'compute_torch_features',

]
__author__ = '陆家立'
__email__ = '996153075@qq.com'
__version__ = '1.0.0'

import json
import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data.data import Data
from tqdm import tqdm

import evolution.feature as feature
import evolution.template as template


# ================================================== 特殊属性与导入 ==================================================


# ================================================== 全局变量 ==================================================

# ================================================== 全局变量 ==================================================


# ================================================== 函数定义 ==================================================
def compute_features(sequences: str, max_len: int, cf: callable):
    """
    计算特征

    :example:
    >>> newCf = feature.esm2_features
    >>> result = compute_features(sequences='TEST', cf=newCf, max_len=5)
    >>> isinstance(result, np.ndarray)
    True

    :param sequences: 数据表格
    :param cf: 计算方法
    :param max_len: 序列最大长度
    :return:
    """
    if not isinstance(sequences, str):
        raise TypeError(f"序列类型 {type(sequences)} 错误, 必须是str类型")
    elif not callable(cf):
        raise TypeError(f"计算方法 {type(cf)} 错误, 必须是函数类型")
    elif not isinstance(max_len, int):
        raise TypeError(f"序列最大长度 {type(max_len)} 错误, 必须是int类型")
    return cf(sequences, max_len=max_len)


def compute_Data(
        cf=None,
        cf_name=None,
        idx=None,
        name=None,
        pdb_path=None,
        sequences=None,
        max_len=None,
        y=None,
):
    """
    计算数据特征

    :example:
    >>> pdbPath = '../exclude/test.pdb'
    >>> Sequences, _ = feature.sequences_features(pdbPath)
    >>> result = compute_Data(
    ...     cf=feature.esm2_features,
    ...     cf_name='ESM2',
    ...     idx=0,
    ...     name='test',
    ...     pdb_path=pdbPath,
    ...     sequences=Sequences,
    ...     max_len=len(Sequences),
    ...     y=1.1,)
    >>> isinstance(result, Data)
    True

    :param cf: 计算方法
    :param cf_name: 方法名
    :param idx: 数据编号(索引)
    :param name: 数据名称
    :param pdb_path: 蛋白PDB路径
    :param sequences: 蛋白单字母序列
    :param max_len: 序列最大长度
    :param y: 数据标签
    :return:
    """
    x = cf(sequences, max_len=max_len)
    assert isinstance(x, np.ndarray), f"特征计算错误: type(x)={type(x)}, x ={x}计算{name}数据的{cf_name}特征时发生错误"
    edge_index = feature.edge_features(pdb_path)
    assert len(edge_index[0]) != 0, f"边数据计算错误: 计算{name}数据的{cf_name}特征时发生错误\nedge_index = {edge_index}"
    return Data(
                index=torch.tensor(idx, dtype=torch.int64),
                name=name,
                x=torch.tensor(x, dtype=torch.float32),
                y=torch.tensor(y, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.int64),
    )


def compute_torch_features(
        df: pd.DataFrame,
        cf: callable,
        max_len: int,
        df_name: str = '',
        cf_name: str = '',
        is_save: bool = True,
        cache: bool = True,
        sf: str = template.PACKAGE_DIR+'/exclude/test.pt',
) -> bool:
    """
    计算图数据特征

    :example:
    >>> from evolution import template
    >>> pdbDir = template.DEFAULT_SAVE_DIR+'/OUT'
    >>> allDataFrame = json.load(open('../exclude/all.json'))
    >>> newCf = feature.esm2_features
    >>> for key, value in allDataFrame.items():
    ...     if key != 'PG16': continue
    ...     newDf = pd.DataFrame(dict(value), columns=value.keys())
    ...     maxLen = max(newDf['Len'])
    ...     result = compute_torch_features(df=newDf.iloc[:3], cf=newCf, max_len=maxLen,
    ...         df_name='PG16', cf_name='ESM2',
    ...         sf='../exclude/test.pt',
    ...         is_save=True, cache=False)
    ...     isinstance(result, bool)
    True

    :param df: 表格数据
    :param cf: 计算方法
    :param max_len: 最大数据长度
    :param sf: 保存文件
    :param df_name: 数据名称
    :param cf_name: 方法名称
    :param is_save: 是否保存数据文件
    :param cache: 是否开启缓存
    :return:
    """
    if not isinstance(df, (pd.DataFrame, pd.Series, dict)):
        raise TypeError(f"数据类型 {type(df)} 错误, 必须是DataFrame或Series类型")
    elif isinstance(df, (dict, pd.Series)):
        data_frame = pd.DataFrame(data=df, columns=list(df.keys()))
    else:
        data_frame = df.copy()
    if not callable(cf):
        raise TypeError(f"计算方法 {type(cf)} 错误, 必须是函数类型")
    if not os.path.exists(os.path.dirname(sf)):
        os.makedirs(os.path.dirname(sf))

    idx_list = []
    torch_list = []
    if is_save and template.is_exists_file(sf):
        try:
            existing_data = torch.load(sf, weights_only=False)
            if not isinstance(existing_data, list):
                existing_data = [existing_data]
            for existing in existing_data:
                print("existing =", existing)
                idx_list.append(existing.index[0])
            torch_list = existing_data
            del existing_data
        except FileNotFoundError:
            pass
    for index, row in tqdm(
            data_frame.iterrows(),
            total=len(data_frame),
            desc=f"{df_name}数据{cf_name}特征计算进度"):
        try:
            if idx_list and int(row["ID"]) in idx_list:
                continue
            else:
                idx_list.append(int(row["ID"]))
            idx = row["ID"]
            name = row["Name"]
            x = torch.tensor(cf(row["X"], max_len=max_len), dtype=torch.float)
            y = torch.tensor([row["Y"]], dtype=torch.float)
            sequences = feature.edge_features(row["PDB"])
            if not sequences[0]:
                raise Exception(f"{name}数据没有序列信息")
            edge_index = torch.tensor(sequences, dtype=torch.long)
            existing_data = None
            torch_data = Data(
                index=[idx],
                name=name,
                x=x,
                y=y,
                edge_index=edge_index)
            if not is_save or not cache:
                torch_list.append(torch_data)
            elif cache:
                try:
                    existing_data = torch.load(sf, weights_only=False)
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data]
                    existing_data.append(torch_data)
                except FileNotFoundError:
                    existing_data = [torch_data]
                torch.save(existing_data, sf)
                del torch_data
        except Exception as e:
            # print("报错数据:", row["X"])
            raise Exception(f"计算{row["Name"]}数据的{cf_name}特征时发生错误: {e}")
        if is_save and cache:
            torch_list = existing_data
        elif is_save:
            torch.save(torch_list, sf)
            # print("保存位置:", os.path.abspath(sf))
    if is_save:
        torch.save(torch_list, sf)
    return True
# ================================================== 函数定义 ==================================================
