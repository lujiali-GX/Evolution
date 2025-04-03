# -*- coding: utf-8 -*-

"""
----
文件名称: preconditioning.py
----
\\
----
模块概述: 数据预处理
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
    - os
    - shutil
    - typing.Union as Union
    - pandas as pd
    - tqdm.tqdm as tqdm
\\
\\
----
使用示例：
----
    - preconditioning.processing_file_name(data_dir=template.RAW_PDB_DIR, out_dir=template.PDB_DIR, del_end_str='_B1.', suffix='.pdb')
    - preconditioning.processing_file(data_path=template.RAW_DATA_DIR+'/PG16.xlsx', pdb_dir=template.PDB_DIR+'/PG16')
    - preconditioning.processing_file_dir(input_dir=template.RAW_DATA_DIR, pdb_dir=template.PDB_DIR, p_type='df')
\\
\\
----    
异常处理：
----
    -
\\
\\
注意事项：
----
    - 在 processing_file_dir 和 processing_file方法中,
    - pdb_dir中的目录名称要与 input_path 的名称以及 input_dir目录中的名称一致
\\
"""

# ================================================== 特殊属性与导入 ==================================================
# __name__
# __doc__
__all__ = [
    # 变量

    # 函数
    'processing_file_name',
    'processing_file',
    'processing_file_dir',
]
__author__ = '陆家立'
__email__ = '996153075@qq.com'
__version__ = '1.0.0'

import os
import shutil
from typing import Union
import pandas as pd
from tqdm import tqdm

from evolution import feature, template


# ================================================== 特殊属性与导入 ==================================================


# ================================================== 全局变量 ==================================================

# ================================================== 全局变量 ==================================================


# ================================================== 函数定义 ==================================================
def processing_file_name(
        data_dir: str, out_dir: str = None,
        del_end_str: str = "_B1.", suffix: str = ".pdb"):
    """
    文件名称重命名

    :example:
    >>> result = processing_file_name(data_dir=template.RAW_PDB_DIR, out_dir=template.PDB_DIR, del_end_str='_B1.', suffix='.pdb')
    >>> isinstance(result, bool)
    True

    :param data_dir:
    :param out_dir:
    :param del_end_str:
    :param suffix:
    :return:
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录{data_dir}不存在")
    elif not len(os.listdir(data_dir)):
        raise FileNotFoundError(f"数据目录{data_dir}为空")
    if out_dir is None:
        out_dir = os.path.join(os.path.abspath(os.path.dirname(data_dir)), 'out')
    else:
        out_dir = os.path.abspath(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for root, dirs, files in os.walk(data_dir):
        if dirs or not files or not root: continue
        dir_name = os.path.basename(root)
        new_out_dir = os.path.join(out_dir, dir_name)
        if not os.path.exists(new_out_dir):
            os.makedirs(new_out_dir)
        for file in tqdm(files, total=len(files), desc="文件重命名进度"):
            if not file.endswith(suffix): continue
            new_file_name = file.replace(del_end_str+suffix, suffix)
            file_path = str(os.path.join(root, file))
            new_file_path = str(os.path.join(new_out_dir, new_file_name))
            shutil.copy(file_path, new_file_path)

    return True


def processing_file(
        data_path: str,
        data_id: str = None,
        pdb_dir: str = "/home/ljl/data/Evolution/pdbDatabase/pdb/PG16",
        name: str = "Virus",
        x_key: str = "Extracted Sequence",
        y_key: str = "log_IC50",
        exception_x: list = None,
        exception_y: list = None,
) -> Union[pd.DataFrame, None]:
    """
    数据文件预处理

    :example:
    >>> result = processing_file(data_path=template.RAW_DATA_DIR+'/PG16.xlsx', pdb_dir=template.PDB_DIR+'/PG16')
    >>> isinstance(result, (pd.DataFrame, None))
    True

    :param data_path: 数据文件路径
    :param data_id: 数据ID列
    :param pdb_dir: PBD 数据目录
    :param name: 数据名称列
    :param x_key: 数据列
    :param y_key: 标签列
    :param exception_x: 异常数据项列表
    :param exception_y: 异常标签项列表

    :return: json数据

    :raise FileNotFoundError: 数据文件不存在
    :raise ValueError: 数据文件格式错误
    """

    data_dir = os.path.dirname(data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件{data_path}不存在")
    if exception_y is None:
        exception_y = ["<", ">"]
    if exception_x is None:
        exception_x = ["*", "#"]
    data_frame = None
    if data_path.endswith(".csv"):
        data_frame = pd.read_csv(data_path)
    elif data_path.endswith(".xlsx"):
        data_frame = pd.read_excel(data_path)
    elif data_path.endswith(".txt"):
        data_frame = pd.read_table(data_path)
    if data_frame is None:
        raise ValueError(f"数据文件 {data_path} 格式错误")
    if x_key not in data_frame.keys():
        # raise ValueError("数据文件列名错误")
        data_frame = None
    if data_frame is None:
        return None
    if pdb_dir is None:
        # print(f"\033[31m警告: PDB 目录未设置, 已使用默认路径生成方式\033[0m")
        pdb_dir = os.path.join(os.path.dirname(data_dir), 'pdb')

    if data_id not in data_frame.keys():
        data_id_list = [i for i in range(len(data_frame))]
        data_frame[data_id] = data_id_list
    if name is None or name not in data_frame.keys():
        data_frame[name] = [name] * len(data_frame)
    if y_key not in data_frame.keys():
        data_frame[y_key] = [0] * len(data_frame)

    if data_frame is None:
        return None
    pdb_list = []
    for index, row in data_frame.iterrows():
        new_x = None
        is_exception_y = False
        for exception in exception_x:
            if exception in str(row[x_key]):
                new_x = row[x_key].replace(exception, "")
        for exception in exception_y:
            if exception in str(row[y_key]):
                data_frame.drop(index, inplace=True)
                is_exception_y = True
                print(f"删除第[{index}]条[{row[name]}]的[{row[y_key]}]数据")
                break
        if new_x is not None and not is_exception_y:
            data_frame.at[index, x_key] = new_x
        if not is_exception_y:
            pdb_path = os.path.join(pdb_dir, f"{row[name]}.pdb")
            pdb_list.append(pdb_path)

    data_frame["PDB"] = pdb_list
    return pd.DataFrame(
        {
            "ID": data_frame[data_id],
            "PDB": data_frame["PDB"],
            "Name": data_frame[name],
            "X": data_frame[x_key],
            "Y": data_frame[y_key],
        },
        columns=["ID", "Name", "X", "Y"],
        index=data_frame.index,
        dtype=None,
        copy=True,
    )


def processing_file_dir(
        input_dir: str,
        pdb_dir: str = template.PDB_DIR,
        p_type: str = "dict"
) -> Union[pd.DataFrame, dict, None]:
    """
    数据目录处理

    :example:
    >>> result = processing_file_dir(input_dir=template.RAW_DATA_DIR, pdb_dir=template.PDB_DIR, p_type='df')
    >>> isinstance(result, pd.DataFrame)
    True

    :param input_dir: 数据表格目录
    :param pdb_dir: PDB 目录
    :param p_type: 输出类型
    :return:
    :raise FileNotFoundError: 文件错误
    :raise TypeError: 类型错误
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"文件错误: 数据目录{input_dir}不存在")
    if pdb_dir is None:
        # print(f"\033[31m警告: PDB 目录未设置, 已使用默认路径生成方式\033[0m")
        pdb_dir = os.path.join(os.path.abspath(os.path.dirname(input_dir)), 'pdb')
    else:
        pdb_dir = os.path.abspath(pdb_dir)
    all_data = pd.DataFrame()
    for file in os.listdir(input_dir):
        file_name = os.path.splitext(file)[0]
        file_path = os.path.join(input_dir, file)
        new_pdb_dir = os.path.join(pdb_dir, file_name)
        df = processing_file(
            data_path=file_path,
            pdb_dir= new_pdb_dir,
            name="Virus",
            x_key="Extracted Sequence",
            y_key="log_IC50",
            exception_x=["*", "#", "-"],
            exception_y=["<", ">"],
        )
        id_list = []
        pdb_list = []
        name_list = []
        len_list = []
        seq_list = []
        label_list = []
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"[{file_name}]数据字典生成进度"):
            pdb_path = os.path.join(new_pdb_dir, row["Name"]+".pdb")
            if not os.path.exists(pdb_path):
                continue
            else:
                seq, seq_len = feature.sequences_features(pdb_path)
                id_list.append(row["ID"])
                pdb_list.append(pdb_path)
                name_list.append(row["Name"])
                len_list.append(seq_len)
                seq_list.append(seq)
                label_list.append(row["Y"])
        if len(pdb_list):
            all_data[file_name] = {
                "ID": id_list,
                "PDB": pdb_list,
                "Name": name_list,
                "Len": len_list,
                "X": seq_list,
                "Y": label_list,
            }
        else:
            print(f"数据目录 {new_pdb_dir} 中没有PDB文件")
    if p_type.lower() in ["df", "dataframe"] and len(all_data):
        all_data = pd.DataFrame(all_data)
    elif len(all_data):
        all_data = all_data.to_dict()
    else:
        raise ValueError("all_data 数据为空")

    return all_data
# ================================================== 函数定义 ==================================================
