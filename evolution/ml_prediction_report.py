# -*- coding: utf-8 -*-

"""
----
文件名称: ml_prediction_report.py
----
\\
----
模块概述: 机器学习模型预测报告
----
\\
----
作   者: ljl (996153075@qq.com)
----
\\
----
创建日期: 2025/4/2
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
    - ml_prediction_report.main()
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

import os
import sys

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from evolution import template, data, models


# ================================================== 特殊属性与导入 ==================================================


# ================================================== 全局变量 ==================================================

# ================================================== 全局变量 ==================================================


# ================================================== 函数定义 ==================================================
def ml_prediction_args():
    """机器学习模型预测外参"""
    usage_str = "-omd 输入机器学习模型数据集目录 -omm 输出机器学习模型目录 -omd 输出机器模型预测报告目录 -l 报告语言"
    print_help = """=================================== 参数说明 ===================================
    短选项	长选项	                    类型	    默认值	                    说明
    -h	    --help	                    bool    False            	        查看此帮助
    -omd	--out_ml_dataset_dir	    str	    template.ML_DATASET_DIR 	输入机器学习模型数据集目录
    -omm	--out_ml_model_dir	    str	    template.ML_MODEL_DIR 	    输出机器学习模型目录
    -omr    --out_ml_report_dir	        str	    template.ML_REPORT_DIR	    输出机器模型预测报告目录
    -l      --language	                str	    'zh'	                    语言[`zh`, `en`]
        =================================== 参数说明 ==================================="""
    epilog = "-omd ./ML_DATASET -omm ./ML_MODEL -omd ./REPORT/ML -l zh"
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
        '-omd', '--out_ml_dataset_dir', 类型=str,
        默认=template.ML_DATASET_DIR, 帮助='输入机器学习模型数据集目录')
    parser.添加参数(
        '-omm', '--out_ml_model_dir', 类型=str,
        默认=template.ML_MODEL_DIR, 帮助='输出机器学习模型目录')
    parser.添加参数(
        '-omr', '--out_ml_report_dir', 类型=str,
        默认=template.ML_REPORT_DIR, 帮助='输出机器模型预测报告目录')
    parser.添加参数(
        '-l', '--language', 类型=str,
        默认='zh', 帮助='语言')
    return parser


def main_args():
    ml_args = ml_prediction_args()
    usage_str = "-omd 输入机器学习模型数据集目录 -omm 输出机器学习模型目录 -omd 输出机器模型预测报告目录 -l 报告语言"
    print_help = """=================================== 参数说明 ===================================
    短选项	长选项	                    类型	    默认值	                    说明
    -h	    --help	                    bool    False            	        查看此帮助
    -omd	--out_ml_dataset_dir	    str	    template.ML_DATASET_DIR 	输入机器学习模型数据集目录
    -omr    --out_ml_report_dir	        str	    template.ML_REPORT_DIR	    输出机器模型预测报告目录
    -l      --language	                str	    'zh'	                    语言[`zh`, `en`]
        =================================== 参数说明 ==================================="""
    epilog = "-omd ./ML_DATASET -omm ./ML_MODEL -omd ./REPORT/ML -l zh"
    parser = template.参数对象(
        程序名称=f'数据处理主函数脚本: {os.path.abspath(__file__)}',
        其他对象=[ml_args],
        使用方法=usage_str,
        补充说明=epilog,
        # 功能描述=print_help,
        冲突处理='resolve',
        帮助选项=False,
    )
    parser.添加参数(
        '-h', '--help', 动作='store_true', 帮助=print_help)
    return parser


def ml_prediction_main(args):
    assert template.is_exists_dir(args.out_ml_dataset_dir), f"\n读取错误: 输入机器学习模型数据集目录不存在 --> {args.out_ml_dataset_dir}"
    template.gen_dir(args.out_ml_report_dir, args.out_ml_model_dir)

    save_all_predict_report_path = os.path.join(args.out_ml_report_dir, "all_ml_predict_report.xlsx")
    for root, dirs, files in os.walk(args.out_ml_dataset_dir):
        if dirs or not files or not root: continue
        data_name: str = os.path.basename(os.path.dirname(root))
        feature_name: str = os.path.basename(root)
        new_save_model_dir = os.path.join(args.out_ml_model_dir, data_name, feature_name)
        new_save_report_dir = os.path.join(args.out_ml_report_dir, data_name, feature_name)
        template.gen_dir(new_save_report_dir, new_save_model_dir)

        data_dict = {}
        for file in files:
            file: str = file
            file_path = str(os.path.join(root, file))
            file_name = file.replace(".npy", "")
            _, _, tag = file_name.split("_")
            data_dict[tag] = np.load(file_path)

        x_list = []
        for x in data_dict["X"]:
            x_list.append(np.mean(np.array(x), axis=0))
        x_array = np.array(x_list)
        # 按列计算平均值
        x_train, x_test, y_train, y_test, = train_test_split(
            x_array, data_dict["Y"],
            test_size=0.2, random_state=42
        )

        save_report_path = os.path.join(new_save_report_dir, f"{data_name}_{feature_name}_report.xlsx")
        for model_name, model_class in tqdm(models.ml.ML_MODEL_DICT.items(), total=len(models.ml.ML_MODEL_DICT), desc=f"[{data_name}]机器学习模型预测进度"):
            save_model_path = os.path.join(new_save_model_dir, f"{data_name}_{feature_name}_{model_name}.joblib")
            model = model_class(
                x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                feature_name=feature_name,
                decimal=3,
                show_report=False,
                metrics_type=args.language,
                )
            report = model.report
            data.save_report(report, save_path=save_all_predict_report_path, language=args.language)
            data.save_report(report, save_path=save_report_path, language=args.language)
            # 保存模型
            joblib.dump(model, save_model_path)
    return True
# ================================================== 函数定义 ==================================================


# ================================================== 类定义 ==================================================

# ================================================== 类定义 ==================================================


# ================================================== 主函数 ==================================================
def main():
    parser = main_args()
    args = parser.parse_args()
    assert ml_prediction_main(args), f"{ml_prediction_args().print_help()}\n读取错误: 输出机器模型预测报告目录失败: {args.out_ml_report_dir}"


# ================================================== 主函数 ==================================================


# ================================================== 程序入口 ==================================================
if __name__ == "__main__":
    main()
# ================================================== 程序入口 ==================================================
