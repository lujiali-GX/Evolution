# -*- coding: utf-8 -*-

"""
----
文件名称: data_processing.py
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
创建日期: 2025/4/1
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
    - data_processing.main()
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

    # 类

    # 主函数
    'main',
]
__author__ = '陆家立'
__email__ = '996153075@qq.com'
__version__ = '1.0.0'

import argparse
import os.path
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from evolution import template, data, compute, feature


# ================================================== 特殊属性与导入 ==================================================


# ================================================== 全局变量 ==================================================

# ================================================== 全局变量 ==================================================


# ================================================== 函数定义 ==================================================
def save_json_args():
    usage_str = "-i/--data_dir 输入xlsx数据目录 -p/--pdb_dir 输入原始PDB目录 -op/--out_pdb_dir 输出PDB目录 -oj/--out_json_dir 输出json目录"
    print_help = """=================================== 参数说明 ===================================
短选项	长选项	    类型	    默认值	                说明
-h	--help	        bool    False          	        查看此帮助
-i	--data_dir	    str	    template.RAW_DATA_DIR	输入原始XLSX目录
-p	--pdb_dir	    str	    template.RAW_PDB_DIR	输入原始PDB目录
-op	--out_pdb_dir	str	    template.PDB_DIR	    输出PDB目录
-oj	--out_json_dir	str	    template.JSON_DIR	    输出JSON目录
=================================== 参数说明 ==================================="""
    epilog = "-i ./XLSX -p ./RAW_PDB -op ./PDB oj ./JSON"
    # 解析命令行参数
    parser = template.参数对象(
        程序名称=f'JSON数据处理脚本: {os.path.abspath(__file__)}',
        使用方法=usage_str,
        补充说明=epilog,
        # 功能描述=print_help,
        帮助类=argparse.RawDescriptionHelpFormatter,
        帮助选项=False,
    )
    parser.添加参数(
        '-h','--help', 动作='store_true',  help=print_help)
    parser.添加参数(
        '-i','--data_dir', 类型=str,
        默认=template.RAW_DATA_DIR, 帮助='输入原始XLSX目录')
    parser.添加参数(
        '-p','--pdb_dir', 类型=str,
        默认=template.RAW_PDB_DIR, 帮助='输入原始PDB目录')
    parser.添加参数(
        '-op', '--out_pdb_dir', 类型=str,
        默认=template.PDB_DIR, 帮助='输出PDB目录')
    parser.添加参数(
        '-oj', '--out_json_dir', 类型=str,
        默认=template.JSON_DIR, help='输出JSON目录')
    return parser


def save_data_args():
    usage_str = "-oj 输入JSON目录 -od 输出原始数据目录"
    print_help = """=================================== 参数说明 ===================================
短选项	长选项	    类型	    默认值	                说明
-h	--help	        bool    False         	        查看此帮助
-oj	--out_json_dir	str	    template.JSON_DIR	    输入JSON目录
-od	--out_data_dir	str	    template.DATA_DIR	    输出原始数据集目录
    =================================== 参数说明 ==================================="""
    epilog = "-oj ./JSON -od ./DATA"
    parser = template.参数对象(
        程序名称=f'保存pt数据: {os.path.abspath(__file__)}',
        使用方法=usage_str,
        补充说明=epilog,
        功能描述=print_help,
        帮助选项=False,

    )
    parser.添加参数(
        '-h','--help', 动作='store_true', 帮助=print_help)
    parser.添加参数(
        '-oj','--out_json_dir', 类型=str,
        默认=template.JSON_DIR, 帮助='输入JSON目录')
    parser.添加参数(
        '-od','--out_data_dir', 类型=str,
        默认=template.DATA_DIR, 帮助='输出原始数据集目录')
    return parser


def save_all_dataset_args():
    """保存所有数据集, 方便进行机器学习"""
    usage_str = "-od 输入原始数据目录 -oas 输出数据集目录"
    print_help = """=================================== 参数说明 ===================================
短选项	长选项	                类型	    默认值	                    说明
-h	    --help	                bool    False            	        查看此帮助
-od	    --out_data_dir	        str	    template.DATA_DIR	        输入原始数据集目录
-oas    --out_all_dataset_dir	str	    template.ALL_DATASET_DIR	输出数据集目录
    =================================== 参数说明 ==================================="""
    epilog = "-od ./DATA -oas ./ALL_DATASET_DIR"
    parser = template.参数对象(
        程序名称=f'保存所有dataset数据: {os.path.abspath(__file__)}',
        使用方法=usage_str,
        补充说明=epilog,
        功能描述=print_help,
        帮助选项=False,
    )
    parser.添加参数(
        '-h','--help', 动作='store_true', 帮助=print_help)
    parser.添加参数(
        '-od','--out_data_dir', 类型=str,
        默认=template.DATA_DIR, 帮助='输入原始数据集目录')
    parser.添加参数(
        '-oas','--out_all_dataset_dir', 类型=str,
        默认=template.ALL_DATASET_DIR, 帮助='输出数据集目录')
    return parser


def save_ml_dataset_args():
    usage_str = "-oas 输入数据集目录 -omd 输出机器学习数据集目录"
    print_help = """=================================== 参数说明 ===================================
    短选项	长选项	                    类型	    默认值	                    说明
    -h	    --help	                    bool    False            	        查看此帮助
    -oas	--out_all_dataset_dir	    str	    template.ALL_DATASET_DIR	输入数据集目录
    -omd    --out_ml_dataset_dir	    str	    template.ML_DATASET_DIR	    输出机器学习数据集目录
        =================================== 参数说明 ==================================="""
    epilog = "-oas ./ALL_DATASET -omd ./ML_DATASET"
    parser = template.参数对象(
        程序名称=f'保存划分dataset数据: {os.path.abspath(__file__)}',
        使用方法=usage_str,
        补充说明=epilog,
        功能描述=print_help,
        帮助选项=False,
    )
    parser.添加参数(
        '-h', '--help', 动作='store_true', 帮助=print_help)
    parser.添加参数(
        '-oas', '--out_all_dataset_dir', 类型=str,
        默认=template.ALL_DATASET_DIR, 帮助='输入数据集目录')
    parser.添加参数(
        '-omd', '--out_ml_dataset_dir', 类型=str,
        默认=template.ML_DATASET_DIR, 帮助='输出机器学习数据集目录')
    return parser


def save_divide_dataset_args():
    usage_str = "-oas 输入数据集目录 -ods 输出深度学习数据集目录"
    print_help = """=================================== 参数说明 ===================================
短选项	长选项	                    类型	    默认值	                    说明
-h	    --help	                    bool    False            	        查看此帮助
-oas	--out_all_dataset_dir	    str	    template.ALL_DATASET_DIR	输入数据集目录
-ods    --out_divide_dataset_dir	str	    template.DATASET_DIR	    输出深度学习数据集目录
    =================================== 参数说明 ==================================="""
    epilog = "-oas ./ALL_DATASET  -ods ./DATASET"
    parser = template.参数对象(
        程序名称=f'保存划分dataset数据: {os.path.abspath(__file__)}',
        使用方法=usage_str,
        补充说明=epilog,
        功能描述=print_help,
        帮助选项=False,
    )
    parser.添加参数(
        '-h','--help', 动作='store_true', 帮助=print_help)
    parser.添加参数(
        '-oas','--out_all_dataset_dir', 类型=str,
        默认=template.ALL_DATASET_DIR, 帮助='输入数据集目录')
    parser.添加参数(
        '-ods','--out_divide_dataset_dir', 类型=str,
        默认=template.DATASET_DIR, 帮助='输出深度学习数据集目录')
    return parser


def main_args():
    json_args = save_json_args()
    data_args = save_data_args()
    all_dataset_args = save_all_dataset_args()
    ml_dataset_args = save_ml_dataset_args()
    dataset_args = save_divide_dataset_args()
    usage_str = "-i 原始XLSX文件目录 -p 原始PDB文件目录 -op 输出PDB目录 -oj 输出JSON文件目录 -od 输出原始数据集目录 -oas 输出数据集目录 -omd 输出机器学习数据集目录 -ods 输出深度学习数据集目录"
    print_help = """=================================== 参数说明 ===================================
    短选项	长选项	                    类型	    默认值	                    说明
    -h	    --help	                    bool    False            	        查看此帮助
    -i	    --data_dir	                str	    template.RAW_DATA_DIR	    输入XLSX数据目录
    -p	    --pdb_dir	                str	    template.RAW_PDB_DIR	    输入原始PDB目录
    -op	    --out_pdb_dir	            str	    template.PDB_DIR	        输出 PDB 目录
    -oj	    --out_json_dir	            str	    template.JSON_DIR	        输出 JSON 目录
    -od	-   -out_data_dir	            str	    template.DATA_DIR	        输出原始数据目录
    -oas	--out_all_dataset_dir	    str	    template.ALL_DATASET_DIR	输出数据集目录
    -omd    --out_ml_dataset_dir	    str	    template.ML_DATASET_DIR	    输出机器学习数据集目录
    -ods	--out_divide_dataset_dir	str	    template.DATASET_DIR        输出深度学习数据集目录  
    =================================== 参数说明 ==================================="""
    epilog = "-i ./XLSX -p ./RAW_PDB -op ./PDB -oj ./JSON -od ./DATA -oas ./ALL_DATASET -omd ./ML_DATASET  -ods ./DATASET"
    parser = template.参数对象(
        程序名称=f'数据处理主函数脚本: {os.path.abspath(__file__)}',
        其他对象=[json_args, data_args, all_dataset_args, ml_dataset_args, dataset_args],
        使用方法=usage_str,
        补充说明=epilog,
        # 功能描述=print_help,
        冲突处理='resolve',
        帮助选项=False,
    )
    parser.添加参数(
        '-h', '--help', 动作='store_true', 帮助=print_help)
    return parser
# ================================================== 函数定义 ==================================================


# ================================================== 类定义 ==================================================

# ================================================== 类定义 ==================================================


# ================================================== 主函数 ==================================================
def save_json_main(args):
    assert os.path.exists(args.data_dir), f"\n读取错误: xlsx数据目录不存在 --> {args.data_dir}"
    assert os.path.exists(args.pdb_dir), f"\n读取错误: PDB 目录不存在 --> {args.pdb_dir}"

    if not os.path.exists(args.out_pdb_dir) or not os.path.isdir(args.out_pdb_dir):
        os.makedirs(args.out_pdb_dir)
    if not os.path.exists(args.out_json_dir) or not os.path.isdir(args.out_json_dir):
        os.makedirs(args.out_json_dir)
    result = data.preconditioning.processing_file_name(
        data_dir=args.pdb_dir,
        out_dir=args.out_pdb_dir,
    )
    assert result

    all_dict = data.preconditioning.processing_file_dir(
        input_dir=args.data_dir,
        pdb_dir=args.out_pdb_dir,
        p_type='dict'
    )
    assert isinstance(all_dict, dict), f"\n读取错误: 数据目录处理失败 --> {args.data_dir}"
    for key, value in all_dict.items():
        json_path = os.path.join(str(args.out_json_dir), f"{key}.json")
        if os.path.exists(json_path):
            continue
        save_json_result = data.save_json_data(value, json_path)
        assert save_json_result, f"\n保存错误: json文件保存失败 --> {json_path}"
        assert not len(os.listdir(args.out_json_dir)), f"\n读取错误: json文件保存失败 --> {json_path}"
    return True


def save_data_main(args):
    assert os.path.exists(args.out_json_dir), f"\n读取错误: json目录不存在 --> {args.out_json_dir}"
    assert len(os.listdir(args.out_json_dir)), f"\n读取错误: json目录为空 --> {args.out_json_dir}"
    if not os.path.exists(args.out_data_dir) or not os.path.isdir(args.out_data_dir):
        os.makedirs(args.out_data_dir)
        
    cf = feature.esm2_features
    cf_name = "ESM2"
    for json_file in os.listdir(args.json_dir):
        json_data_name = json_file.replace(".json", "")
        json_path = os.path.join(args.json_dir, json_file)
        json_dict = data.get_json_data(path=json_path)
        max_len = max(json_dict["Len"])

        new_out_data_dir = os.path.join(args.out_data_dir, json_data_name, cf_name)
        if not os.path.exists(new_out_data_dir) or not os.path.isdir(new_out_data_dir):
            os.makedirs(new_out_data_dir)
        json_data_frame = pd.DataFrame(data=json_dict, columns=list(json_dict.keys()))

        for index, row in tqdm(json_data_frame.iterrows(), total=len(json_data_frame), desc=f"[{json_data_name}]数据处理进度"):
            save_data_path = os.path.join(new_out_data_dir, f"{row['Name']}.pt")
            torch_data = compute.compute_Data(
                cf=cf,
                cf_name=cf_name,
                idx=row["ID"],
                name=row["Name"],
                pdb_path=row["PDB"],
                sequences=row["X"],
                max_len=max_len,
                y=row["Y"],
            )
            assert isinstance(torch_data, Data)
            torch.save(torch_data, save_data_path)
    return True


def save_all_dataset_main(args):
    assert os.path.exists(args.out_data_dir), f"\n读取错误: pt数据目录不存在 --> {args.out_data_dir}"
    template.gen_dir(args.out_all_dataset_dir)
    print(f"存储数据集: {args.out_all_dataset_dir}")
    for root, dirs, files in os.walk(args.out_data_dir):
        if dirs or not files or not root: continue
        data_name: str = os.path.basename(os.path.dirname(root))
        feature_name: str = os.path.basename(root)
        new_out_all_dataset_dir: str = os.path.join(args.out_all_dataset_dir, data_name)
        new_out_all_dataset_path = os.path.join(new_out_all_dataset_dir, f"{data_name}_{feature_name}.pt")
        template.gen_dir(new_out_all_dataset_dir)
        data_list = []
        for file in tqdm(files, total=len(files), desc=f"[{data_name}]数据提取进度"):
            file_path = str(os.path.join(root, file))
            torch_data = torch.load(file_path, weights_only=False)
            data_list.append(torch_data)
        data.save_torch_data(data_list, new_out_all_dataset_path)
        assert template.is_exists_file(new_out_all_dataset_path), f"\n保存错误: pt文件保存失败 --> {new_out_all_dataset_path}"
        return True


def save_ml_dataset_main(args):
    assert template.is_exists_dir(args.out_all_dataset_dir), f"\n读取错误: 数据集目录不存在 --> {args.out_all_dataset_dir}"
    template.gen_dir(args.out_ml_dataset_dir)

    for root, dirs, files in os.walk(args.out_all_dataset_dir):
        if dirs or not files or not root: continue
        data_name: str = os.path.basename(os.path.dirname(root))
        feature_name: str = os.path.basename(root)
        for file in files:
            file: str = file
            file_path = str(os.path.join(root, file))
            file_name = file.replace(".pt", "")
            new_ml_dataset_dir = os.path.join(args.out_ml_dataset_dir, data_name, feature_name)
            template.gen_dir(new_ml_dataset_dir)
            save_x_ml_dataset_path = os.path.join(new_ml_dataset_dir, f"{file_name}_X.npy")
            save_y_ml_dataset_path = os.path.join(new_ml_dataset_dir, f"{file_name}_Y.npy")

            torch_data = torch.load(file_path, weights_only=False)
            x_data_array = np.array([torch_data[0].x.numpy()])
            y_data_array = np.array([torch_data[0].y.numpy()])
            # for data_ in torch_data[1:]:
            for data_ in tqdm(torch_data[1:], total=len(torch_data[1:]), desc=f"[{data_name}]机器学习数据集提取进度"):
                x_data_array = np.concatenate((x_data_array, np.array([data_.x.numpy()])), axis=0)
                y_data_array = np.concatenate((y_data_array, np.array([data_.y.numpy()])), axis=0)

            np.save(save_x_ml_dataset_path, x_data_array)
            np.save(save_y_ml_dataset_path, y_data_array)
            assert template.is_exists_file(save_x_ml_dataset_path), f"\n保存错误: x数据集保存失败 --> {save_x_ml_dataset_path}"
            assert template.is_exists_file(save_y_ml_dataset_path), f"\n保存错误: y数据集保存失败 --> {save_y_ml_dataset_path}"
    return True



def save_divide_dataset_main(args):
    assert template.is_exists_dir(args.out_all_dataset_dir), f"\n读取错误: 数据集目录不存在 --> {args.out_all_dataset_dir}"
    template.gen_dir(args.out_divide_dataset_dir)

    for root, dirs, files in os.walk(args.out_all_dataset_dir):
        if dirs or not files or not root: continue
        data_name: str = os.path.basename(os.path.dirname(root))
        for file in tqdm(files, total=len(files), desc=f"[{data_name}]深度学习模型数据集划分进度"):
            file: str = file
            file_path = str(os.path.join(root, file))
            file_name = file.replace(".pt", "")
            data_name, feature_name = file_name.split("_")

            new_out_divide_dataset_dir: str = os.path.join(args.out_divide_dataset_dir, data_name, feature_name)
            template.gen_dir(new_out_divide_dataset_dir)

            torch_data = torch.load(file_path, weights_only=False)
            save_train_path = os.path.join(new_out_divide_dataset_dir, f"{file_name}_train.pt")
            save_val_path = os.path.join(new_out_divide_dataset_dir, f"{file_name}_val.pt")
            save_test_path = os.path.join(new_out_divide_dataset_dir, f"{file_name}_test.pt")
            train_dataset, val_dataset, test_dataset = data.divide_data_set(
                data=torch_data,
                ratio=(0.70, 0.15, 0.15), device=torch.device("cpu"),)
            assert train_dataset and val_dataset and test_dataset, f"\n读取错误: 数据集划分失败 --> {data_name}"

            torch.save(train_dataset, save_train_path)
            torch.save(val_dataset, save_val_path)
            torch.save(test_dataset, save_test_path)
            print(len(train_dataset))
            print(len(val_dataset))
            print(len(test_dataset))
            assert len(os.listdir(new_out_divide_dataset_dir))!=0, f"{args.print_help()}\n读取错误: 数据集保存失败 --> {new_out_divide_dataset_dir}"
    return True


def main():
    parser = main_args()
    args = parser.parse_args()
    if not save_json_main(args): sys.exit(f"{save_json_args().print_help()}\n读取错误: 输出JSON目录失败 --> {args.out_json_dir}")
    if not save_data_main(args): sys.exit(f"{save_data_args().print_help()}\n读取错误: 输出原始数据集目录失败 --> {args.out_data_dir}")
    if not save_all_dataset_main(args): sys.exit(f"{save_all_dataset_args().print_help()}\n读取错误: 输出数据集目录失败 --> {args.out_all_dataset_dir}")
    if not save_ml_dataset_main(args): sys.exit(f"{save_ml_dataset_args().print_help()}\n读取错误: 输出机器学习模型数据集目录失败 --> {args.out_ml_dataset_dir}")
    if not save_divide_dataset_main(args): sys.exit(f"{save_divide_dataset_args().print_help()}\n读取错误: 输出深度学习模型数据集目录失败 --> {args.out_divide_dataset_dir}")
# ================================================== 主函数 ==================================================


# ================================================== 程序入口 ==================================================
if __name__ == "__main__":
    main()
# ================================================== 程序入口 ==================================================
