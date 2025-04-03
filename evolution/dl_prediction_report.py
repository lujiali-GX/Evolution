# -*- coding: utf-8 -*-

"""
----
文件名称: dl_prediction_report.py
----
\\
----
模块概述: 深度学习模型预测
----
\\
----
作   者: ljl (996153075@qq.com)
----
\\
----
创建日期: 2025/4/3
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
    - dl_prediction_report.main()
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

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from evolution import template, models, data


# ================================================== 特殊属性与导入 ==================================================


# ================================================== 全局变量 ==================================================

# ================================================== 全局变量 ==================================================


# ================================================== 函数定义 ==================================================
def dl_prediction_args():
    """机器学习模型预测外参"""
    usage_str = "-odd 输入深度学习模型数据集目录 -odm 输出深度学习模型保存目录 -odr 输出深度学习模型预测报告目录 -l 报告语言"
    print_help = """=================================== 参数说明 ===================================
    短选项	长选项	                    类型	    默认值	                    说明
    -h	    --help	                    bool    False            	        查看此帮助
    -odd	--out_dl_dataset_dir	    str	    template.DATASET_DIR 	    输入深度学习模型数据集目录
    -odm	--out_dl_model_dir  	    str	    template.MODEL_DIR 	        输出深度学习模型保存目录
    -odr    --out_dl_report_dir	        str	    template.DL_REPORT_DIR	    输出深度学习模型预测报告目录
    -l      --language	                str	    'zh'	                    语言
        =================================== 参数说明 ==================================="""
    epilog = "-odd ./DL_DATASET -odm ./MODEL_DIR -odr ./REPORT/DL -l zh"
    parser = template.参数对象(
        程序名称=f'深度学习模型预测报告生成器: {os.path.abspath(__file__)}',
        使用方法=usage_str,
        补充说明=epilog,
        功能描述=print_help,
        帮助选项=False,
    )
    parser.添加参数(
        '-h', '--help', 动作='store_true', 帮助=print_help)
    parser.添加参数(
        '-odd', '--out_dl_dataset_dir', 类型=str,
        默认=template.DATASET_DIR, 帮助='输入深度学习模型数据集目录')
    parser.添加参数(
        '-odm', '--out_dl_model_dir', 类型=str,
        默认=template.MODEL_DIR, 帮助='输出深度学习模型保存目录')
    parser.添加参数(
        '-odr', '--out_dl_report_dir', 类型=str,
        默认=template.DL_REPORT_DIR, 帮助='输出深度学习模型预测报告目录')
    parser.添加参数(
        '-d', '--device', 类型=str,
        默认=template.DEVICE, 帮助='计算设备')
    parser.添加参数(
        '-l', '--language', 类型=str,
        默认='zh', 帮助='语言')
    return parser


def main_args():
    dl_args = dl_prediction_args()
    usage_str = "-odd 输入深度学习模型数据集目录 -odr 输出深度学习模型预测报告目录"
    print_help = """=================================== 参数说明 ===================================
    短选项	长选项	                    类型	    默认值	                    说明
    -h	    --help	                    bool    False            	        查看此帮助
    -odd	--out_dl_dataset_dir	    str	    template.DATASET_DIR 	    输入深度学习模型数据集目录
    -odr    --out_dl_report_dir	        str	    template.DL_REPORT_DIR	    输出深度学习模型预测报告目录
    -d      --device	                str	    template.DEVICE     	    计算设备
    -l      --language	                str	    'zh'	                    语言
        =================================== 参数说明 ==================================="""
    epilog = "-odd ./DL_DATASET -odr ./REPORT/DL -l"
    parser = template.参数对象(
        程序名称=f'数据处理主函数脚本: {os.path.abspath(__file__)}',
        其他对象=[dl_args],
        使用方法=usage_str,
        补充说明=epilog,
        # 功能描述=print_help,
        冲突处理='resolve',
        帮助选项=False,
    )
    parser.添加参数(
        '-h', '--help', 动作='store_true', 帮助=print_help)
    return parser


def train(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
):
    model.to(device)
    model.train()
    total_loss = 0
    y_true = []
    y_pre = []
    for torch_data in train_loader:
        torch_data = torch_data.to(device)
        optimizer.zero_grad()
        out = model(torch_data)
        loss = criterion(out[0], torch_data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        y_true.append(torch_data.y.item())
        y_pre.append(out.item())
    return np.array(y_true), np.array(y_pre), total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    # 验证模型
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pre = []
    with torch.no_grad():
        for torch_data in val_loader:
            torch_data = torch_data.to(device)
            out = model(torch_data)
            loss = criterion(out[0], torch_data.y)
            total_loss += loss.item()
            y_true.append(torch_data.y.item())
            y_pre.append(out.item())

    return np.array(y_true), np.array(y_pre), total_loss / len(val_loader)


def predict(
        model,
        test_loader,
        device,
):
    model.eval()  # 设置模型为评估模式
    y_true = []
    y_pre = []
    with torch.no_grad():  # 禁用梯度计算以节省内存
        for torch_data in test_loader:
            torch_data = torch_data.to(device)
            out = model(torch_data)
            y_true.append(torch_data.y.item())
            y_pre.append(out.item())

    return np.array(y_true), np.array(y_pre)


# 简单的可解释性分析：计算每个节点特征的重要性
def interpretability_analysis(model, torch_data, device):
    model.eval()
    torch_data = torch_data.to(device)
    x, edge_index = torch_data.x, torch_data.edge_index
    # 记录原始预测值
    original_out = model(data).item()
    node_importances = []
    for i in range(x.size(0)):
        # 复制一份节点特征
        x_modified = x.clone()
        # 将第 i 个节点的特征置为 0
        x_modified[i] = 0
        # 创建修改后的 Data 对象
        data_modified = Data(x=x_modified, edge_index=edge_index, y=torch_data.y)
        # 计算修改后的预测值
        modified_out = model(data_modified).item()
        # 计算重要性得分，即预测值的变化
        importance = abs(original_out - modified_out)
        node_importances.append(importance)
    return node_importances
# ================================================== 函数定义 ==================================================


# ================================================== 类定义 ==================================================

# ================================================== 类定义 ==================================================


# ================================================== 主函数 ==================================================
def dl_prediction_main(arg, sup_args: template.SuperParameters):
    parser = dl_prediction_args()
    args = parser.parse_args()

    assert template.is_exists_dir(args.out_dl_dataset_dir), f"\n读取错误: 输入深度学习模型数据集目录不存在 --> {args.out_dl_dataset_dir}"
    assert len(os.listdir(args.out_dl_dataset_dir)), f"\n读取错误: 输入深度学习模型数据集目录为空 --> {args.out_dl_dataset_dir}"
    template.gen_dir(args.out_dl_report_dir)
    template.gen_dir(args.out_dl_model_dir)

    if args.language.lower() in ['zh', 'chinese','cn']:
        report_dict = template.ZH_REPORT_DICT
        report_df = template.ZH_REPORT_DATA_FRAME
    elif args.language.lower() in ['en', 'english']:
        report_dict = template.EN_REPORT_DICT
        report_df = template.EN_REPORT_DATA_FRAME
    else:
        report_dict = template.ZH_REPORT_DICT
        report_df = template.ZH_REPORT_DATA_FRAME

    save_all_train_report_path = os.path.join(args.out_dl_report_dir, "all_train_report.xlsx")
    save_all_val_report_path = os.path.join(args.out_dl_report_dir, "all_val_report.xlsx")
    save_all_test_report_path = os.path.join(args.out_dl_report_dir, "all_test_report.xlsx")
    for root, dirs, files in os.walk(args.out_dl_dataset_dir):
        if dirs or not files or not root: continue
        data_name: str = os.path.basename(os.path.dirname(root))
        feature_name: str = os.path.basename(root)
        new_save_report_dir = os.path.join(args.out_dl_report_dir, data_name, feature_name)
        new_save_model_dir = os.path.join(args.out_dl_model_dir, data_name, feature_name)
        template.gen_dir(new_save_report_dir, new_save_model_dir)
        data_dict = {}
        for file in files:
            file: str = file
            file_path = str(os.path.join(root, file))
            file_name = file.split(".")[0]
            _, _, tag = file_name.split("_")
            data_dict[tag] = torch.load(file_path, weights_only=False)

        train_loader = DataLoader(data_dict['train'], sup_args.batch_size, shuffle=True)
        val_loader = DataLoader(data_dict['val'], sup_args.batch_size, shuffle=True)
        test_loader = DataLoader(data_dict['test'], sup_args.batch_size, shuffle=False)
        for conv_name, conv_cls in models.dl.DL_CONV_DICT.items():
            save_train_report_path = os.path.join(new_save_report_dir, f"{data_name}_{feature_name}_{conv_name}_train.xlsx")
            save_val_report_path = os.path.join(new_save_report_dir, f"{data_name}_{feature_name}_{conv_name}_val.xlsx")
            save_test_report_path = os.path.join(new_save_report_dir, f"{data_name}_{feature_name}_{conv_name}_test.xlsx")
            save_model_path = os.path.join(new_save_model_dir, f"{data_name}_{feature_name}_{conv_name}.pth")
            sup_args.conv = conv_cls
            sup_args.conv_name = conv_name
            if not template.is_exists_file(save_model_path):
                train_model = sup_args.model(sup_args)
            else:
                train_model = torch.load(save_model_path, weights_only=False)
            criterion = sup_args.criterion()  # 衡量差异
            optimizer = sup_args.optimizer(train_model.parameters(), lr=sup_args.lr)  # 学习率
            scheduler = sup_args.scheduler(optimizer, step_size=sup_args.step_size, gamma=sup_args.gamma)  # 学习率调整

            # 初始化早停对象
            early_stopping = models.dl.EarlyStopping(patience=sup_args.max_inv_epoch, verbose=True, save_path=save_model_path)
            current_r2 = -999
            current_val_loss = 999
            for epoch in tqdm(range(sup_args.epochs), total=sup_args.epochs,
                              desc=f"[{data_name}]数据[{sup_args.model_name}]模型[{conv_name}]核训练进度"):

                train_y_true, train_y_pred, train_loss = train(
                    model=train_model,
                    train_loader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=sup_args.device,
                )
                val_y_true, val_y_pred, val_loss = validate(
                    model=train_model,
                    val_loader=val_loader,
                    criterion=criterion,
                    device=args.device,
                )
                scheduler.step()
                r2 = models.model_metrics.compute_R2(y_true=val_y_true, y_pred=val_y_pred)
                train_report = models.model_metrics.GET_REGRESSION_METRICS(
                    y_true=train_y_true,
                    y_pred=train_y_pred,
                    current_epoch=epoch+1,
                    feature_name=feature_name,
                    model_name=conv_name,
                    decimal=sup_args.decimal,
                    report_dict=report_dict,
                    report_df=report_df,
                )
                data.save_report(train_report, save_all_train_report_path)
                data.save_report(train_report, save_train_report_path)
                val_report = models.model_metrics.GET_REGRESSION_METRICS(
                    y_true=val_y_true,
                    y_pred=val_y_pred,
                    current_epoch=epoch+1,
                    feature_name=feature_name,
                    model_name=conv_name,
                    decimal=sup_args.decimal,
                    report_dict=report_dict,
                    report_df=report_df,
                )
                data.save_report(val_report, save_all_val_report_path)
                data.save_report(val_report, save_val_report_path)
                early_stopping(val_loss, train_model)
                # if r2 > current_r2:
                if val_loss < current_val_loss:
                    current_val_loss = val_loss
                    current_r2 = r2
                    torch.save(train_model, os.path.join(new_save_model_dir, save_model_path))
                    test_y_true, test_y_pre = predict(train_model, test_loader, sup_args.device)
                    print(f"""
                    当前轮次: {epoch+1}
                    当前模型: {conv_name}
                    当前学习率: {optimizer.param_groups[0]['lr']}
                    训练损失: {train_loss}
                    验证损失: {val_loss}
                    验证R2: {r2}
                    预测R2: {models.model_metrics.compute_R2(y_true=test_y_true, y_pred=test_y_pre)}
                    """)
                    test_report = models.model_metrics.GET_REGRESSION_METRICS(
                        y_true=test_y_true,
                        y_pred=test_y_pre,
                        current_epoch=epoch + 1,
                        feature_name=feature_name,
                        model_name=conv_name,
                        decimal=sup_args.decimal,
                        report_dict=report_dict,
                        report_df=report_df,
                    )
                    data.save_report(test_report, save_all_test_report_path)
                    data.save_report(test_report, save_test_report_path)
    return True


def main():
    parser = main_args()
    args = parser.parse_args()

    sup_args = template.SuperParameters(
        gamma=0.1,
        cls_hidden=20,
    )
    sup_args.device = args.device
    sup_args.model = models.dl.ProteinCNN
    sup_args.model_name = "ProteinCNN"

    assert dl_prediction_main(args, sup_args), f"{dl_prediction_args().print_help()}\n读取错误: 输出深度学习模型预测报告失败: {args.out_dl_model_dir} {args.out_dl_report_dir}"


# ================================================== 主函数 ==================================================


# ================================================== 程序入口 ==================================================
if __name__ == "__main__":
    main()
# ================================================== 程序入口 ==================================================
