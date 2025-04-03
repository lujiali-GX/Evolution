# -*- coding: utf-8 -*-

"""
----
文件名称: __init__.py.py
----
\\
----
模块概述: 全局预设
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
    - argparse
    - os
    - sys
    - typing.Any as Any
    - typing.Union as Union
    - typing.Tuple as Tuple

    - torch
    - pandas as pd
    - torch.nn as nn
    - torch.Tensor as Tensor
    - torch.nn.functional as F
    - torch_geometric.nn.GCNConv as GCNConv
    - torch_geometric.nn.global_mean_pool as global_mean_pool
\\
\\
----
使用示例：
----
    - template.TEMPLATE_DIR -- 模板所在目录
    - template.TEMP_DIR -- 临时目录
    - template.RAW_DATA_DIR -- 原始 xlsx 数据目录
    - template.RAW_PDB_DIR -- 原始 PDB 数据目录
    - template.DEFAULT_SAVE_DIR -- 默认保存位置
    - template.ESM2_DIR -- ESM2 模型目录
    - template.EMS2_MODEL -- ESM2 预训练模型
    - template.EMS2_TOKENIZER -- ESM2 分词器
    - template.PROTTRANS_DIR -- PROTTRANS 模型目录
    - template.PROTTRANS_MODEL -- PROTTRANS 预训练模型
    - template.PROTTRANS_TOKENIZER -- PROTTRANS 分词器
    - template.ML_MODEL_DICT -- 机器学习模型字典
    - models.dl.DL_CONV_DICT -- 深度学习模型字典
    - template.ZH_REPORT_DICT -- 报告字典(英文)
    - template.EN_REPORT_DICT -- 报告字典(英文)
    - template.ZH_REPORT_DATA_FRAME -- 报告数据表(中文)
    - template.EN_REPORT_DATA_FRAME -- 报告数据表(英文)
    - template.AMINO_ACIDS  -- 氨基酸字符串
    - template.SuperParameters() -- 超参
\\
\\
----    
异常处理：
----
    -

\\
----
注意事项：
----
    -
\\
"""

# ================================================== 特殊属性与导入 ==================================================
__all__ = [
    # 全局变量
    "DEVICE",

    "PACKAGE_DIR",
    "TEMPLATE_DIR",
    "TEMP_DIR",

    "RAW_DATA_DIR",
    "RAW_PDB_DIR",

    "DEFAULT_SAVE_DIR",
    "PDB_DIR",
    "JSON_DIR",
    "FEATURE_DIR",
    "DATA_DIR",
    "ALL_DATASET_DIR",
    "DATASET_DIR",
    "ML_DATASET_DIR",

    "MODEL_DIR",
    "REPORT_DIR",
    "ML_REPORT_DIR",
    "DL_REPORT_DIR",

    "ESM2_DIR",
    "PROTTRANS_DIR",
    "ESM2_MODEL",
    "EMS2_TOKENIZER",
    "PROTTRANS_MODEL",
    "PROTTRANS_TOKENIZER",

    "ZH_REPORT_DICT",
    "EN_REPORT_DICT",
    "ZH_REPORT_DATA_FRAME",
    "EN_REPORT_DATA_FRAME",

    "AMINO_ACIDS",
    "AMINO_ACIDS_DICT",

    # 类定义
    "参数对象",
    "SuperParameters",
]
# __name__
# __doc__
__author__ = "陆家立"
__email__ = "996153075@qq.com"
__version__ = "1.0.0"

import argparse
import os
import sys
from typing import Any, Union, Tuple

import torch
import pandas as pd
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

import evolution
# ================================================== 特殊属性与导入 ==================================================


# ================================================== 全局变量 ==================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
"""设备"""

PROTTRANS_DIR = "/home/ljl/models/prot_t5_xl_uniref50"
"""prottrans预训练模型目录"""
ESM2_DIR = "/home/ljl/models/ESM2"
"""ESM2 预训练模型目录"""
ESM2_MODEL = os.path.join(ESM2_DIR, "esm2_t33_650M_UR50D.pt")
"""ESM2 预训练模型"""
EMS2_TOKENIZER = os.path.join(ESM2_DIR, "tokenizer_config.json")
"""ESM2 分词器"""
PROTTRANS_TOKENIZER = os.path.join(PROTTRANS_DIR, "tokenizer_config.json")
"""prottrans 分词器"""
PROTTRANS_MODEL = os.path.join(PROTTRANS_DIR, "pytorch_model.bin")
"""prottrans 预训练模型"""
PACKAGE_DIR = os.path.dirname(os.path.abspath(evolution.__file__))
"""包路径"""
TEMPLATE_DIR = os.path.join(PACKAGE_DIR, "template")
"""模板所在目录"""
TEMP_DIR = os.path.join(os.path.dirname(PACKAGE_DIR), "temp")
"""读取数据所在目录"""

DEFAULT_SAVE_DIR = "/home/ljl/data/Evolution/pdbDatabase"
"""本地默认保存位置"""
RAW_DATA_DIR = os.path.join(DEFAULT_SAVE_DIR, "XLSX")
"""原始数据表格目录: xlsx 文件目录"""
RAW_PDB_DIR = os.path.join(DEFAULT_SAVE_DIR, "RAW_PDB")
"""原始 PDB 数据目录: pdb 文件目录"""
PDB_DIR = os.path.join(DEFAULT_SAVE_DIR, "PDB")
JSON_DIR = os.path.join(DEFAULT_SAVE_DIR, "JSON")
FEATURE_DIR = os.path.join(DEFAULT_SAVE_DIR, "FEATURE")
DATA_DIR = os.path.join(DEFAULT_SAVE_DIR, "DATA")
ALL_DATASET_DIR = os.path.join(DEFAULT_SAVE_DIR, "ALL_DATASET")
DATASET_DIR = os.path.join(DEFAULT_SAVE_DIR, "DATASET")
ML_MODEL_DIR = os.path.join(DEFAULT_SAVE_DIR, "ML_MODEL")
ML_DATASET_DIR = os.path.join(DEFAULT_SAVE_DIR, "ML_DATASET")

MODEL_DIR = os.path.join(DEFAULT_SAVE_DIR, "MODEL")
REPORT_DIR = os.path.join(DEFAULT_SAVE_DIR, "REPORT")
ML_REPORT_DIR = os.path.join(REPORT_DIR, "ML")
DL_REPORT_DIR = os.path.join(REPORT_DIR, "DL")


ZH_REPORT_DICT = {
    "训练轮次[Epoch]": "训练轮次[Epoch]",
    "数据特征[Feature]": "数据特征[Feature]",
    "模型名称[Model]": "模型名称[Model]",
    "平均绝对误差[MAE]": "平均绝对误差[MAE]",
    "均方根误差[RMSE]": "均方根误差[RMSE]",
    "均方误差[MSE]": "均方误差[MSE]",
    "决定系数[R²]": "决定系数[R²]",
    "皮尔逊相关系数[PCC]": "皮尔逊相关系数[PCC]",
    "斯皮尔曼等级相关系数[Spearman]": "斯皮尔曼等级相关系数[Spearman]",
    "肯德尔tau-b系数[KendallTau]": "肯德尔tau-b系数[KendallTau]",
}
"""
报告字典(中文):   \n
训练轮次[Epoch]: 训练轮次[Epoch]    \n
数据特征[Feature]: 数据特征[Feature]    \n
模型名称[Model]: 模型名称[Model]    \n
平均绝对误差[MAE]: 平均绝对误差[MAE]    \n
均方根误差[RMSE]: 均方根误差[RMSE]    \n
决定系数[R²]:   决定系数[R²]    \n
皮尔逊相关系数[PCC]:   皮尔逊相关系数[PCC]    \n
斯皮尔曼等级相关系数[Spearman]:   斯皮尔曼等级相关系数[Spearman]    \n
肯德尔tau-b系数[KendallTau]:   肯德尔tau-b系数[KendallTau]    \n
"""
EN_REPORT_DICT = {
    "训练轮次[Epoch]": "Epoch",
    "数据特征[Feature]": "Feature",
    "模型名称[Model]": "Model",
    "平均绝对误差[MAE]": "MAE",
    "均方根误差[RMSE]": "RMSE",
    "均方误差[MSE]": "MSE",
    "决定系数[R²]": "R²",
    "皮尔逊相关系数[PCC]": "PCC",
    "斯皮尔曼等级相关系数[Spearman]": "Spearman",
    "肯德尔tau-b系数[KendallTau]": "KendallTau",
}
"""
报告字典(英文):   \n
训练轮次[Epoch]: Epoch    \n
数据特征[Feature]: Feature    \n
模型名称[Model]: Model    \n
平均绝对误差[MAE]: MAE    \n
均方根误差[RMSE]: RMSE    \n
决定系数[R²]:   R²    \n
皮尔逊相关系数[PCC]:   PCC   \n
斯皮尔曼等级相关系数[Spearman]:   Spearman    \n
肯德尔tau-b系数[KendallTau]:   KendallTau    \n
"""
ZH_REPORT_DATA_FRAME = pd.DataFrame(
    columns=list(ZH_REPORT_DICT.values())
)
"""报告数据表(中文)"""
EN_REPORT_DATA_FRAME = pd.DataFrame(
    columns=list(EN_REPORT_DICT.values())
)
"""报告数据表(英文)"""
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
"""氨基酸字符串"""
AMINO_ACIDS_DICT = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}
"""氨基酸字典"""
# ================================================== 全局变量 ==================================================


# ================================================== 函数定义 ==================================================
def is_exists_file(*args):
    for i in args:
        if not os.path.exists(i) or not os.path.isfile(i):
            return False
    return True


def is_exists_dir(*args):
    for i in args:
        if not os.path.exists(i) or not os.path.isdir(i):
            return False
    return True


def gen_dir(*args):
    for i in args:
        if not is_exists_dir(i):
            os.makedirs(i)
# ================================================== 函数定义 ==================================================


# ================================================== 类定义 ==================================================
class 参数对象(argparse.ArgumentParser):
    args = argparse
    def __init__(
            self,
            程序名称=None,
            使用方法=None,
            功能描述=None,
            补充说明=None,
            其他对象=None,
            帮助类=argparse.HelpFormatter,
            参数前缀='-',
            读取文件前缀='@',
            参数默认值=None,
            冲突处理="error",
            帮助选项=True,
            缩写选项=True,
            异常退出=True,

                 ):
        """
        外部参数设置
        动作:
            1. store: input >>> Namespace
            2. store_const: const >>> Namespace
            3. store_true: True >>> Namespace
            4. store_false: False >>> Namespace
            5. append: input1 >>> Namespace, input2 >>> Namespace
            6. append_const: const >>> Namespace, const >>> Namespace
            7. count: Namespace, Namespace, Namespace >>> 3
            8. help: print_help()
            9. version: 版本信息

        :param 程序名称:[prog]: 名称
        :param 使用方法:[usage]: 使用方法
        :param 功能描述:[description]: 功能描述
        :param 补充说明:[epilog]: 补充说明(帮助结尾处)
        :param 其他对象:[parents]: 其他 ArgumentParser 外参对象
        :param 帮助类:[formatter_class}: 帮助信息的格式化类
        :param 参数前缀:[prefix_chars}: 参数前缀, 默认`-`
        :param 读取文件前缀:[fromfile_prefix_chars]: 读取文件前缀, 默认`@`
        :param 参数默认值:[argument_default]: 所有参数默认值, 默认 None
        :param 冲突处理:[conflict_handler]: 处理参数冲突方式, 默认`error`抛出错误, 可选[`error`, `resolve`]
        :param 帮助选项:[add_help]: 是否自动添加-h/--help选项
        :param 缩写选项:[allow_abbrev]: 是否允许缩写长选项
        :param 异常退出:[exit_on_error]: 遇到错误是否退出
        """
        super(参数对象, self).__init__(
            prog=程序名称,
            usage=使用方法,
            description=功能描述,
            epilog=补充说明,
            parents=其他对象 if 其他对象 is not None else [],
            formatter_class=帮助类,
            prefix_chars=参数前缀,
            fromfile_prefix_chars=读取文件前缀,
            argument_default=参数默认值,
            conflict_handler=冲突处理,
            add_help=帮助选项,
            allow_abbrev=缩写选项,
            exit_on_error=异常退出,
        )

    def 添加参数(
            self,
            *名称或标志,
            动作=None,
            参数数量=None,
            常量值=None,
            默认=None,
            类型=None,
            可选=None,
            必填=None,
            帮助=None,
            元变量名=None,
            目标属性名=None,
            版本=None,
            action=None,
            nargs=None,
            const=None,
            default=None,
            type=None,
            choices=None,
            required=None,
            help=None,
            metavar=None,
            dest=None,
            version=None,
            **其他关键字参数
    ) ->  argparse.Action:
        """
        添加参数

        :param 名称或标志:[*name_or_flags]: 这是一个可变参数，可传入一个或多个字符串。这些字符串代表命令行参数的名称或标志，例如 '-f'、'--file'。其中以单横线开头的为短标志，以双横线开头的为长标志。
        :param 动作:[action]:
            666:
            该参数可指定在解析到对应参数时要执行的动作。
            它既可以是一个字符串，代表预定义的动作（如 'store'、'store_true'、'store_false' 等），也可以是一个自定义的 Action 类的类型。例如 'store' 会将参数值存储起来，'store_true' 会在遇到该参数时将对应属性设为 True。
        :param 参数数量:[nargs]: 用于指定该参数需要接收的参数值的数量。可以是一个整数，表示固定的参数值数量；也可以是一些特殊字符串，如 '?' 表示参数可选，'*' 表示可接收零个或多个参数值，'+' 表示至少接收一个参数值。
        :param 常量值:[const]: 当 action 为某些特定值（如 'store_const'、'append_const'）时，该参数指定要存储的常量值。
        :param 默认:[default]: 若命令行中未提供该参数，就会使用此默认值。
        :param 类型:[type]: 此参数用于指定参数值的类型，它可以是一个函数，该函数接收一个字符串并返回特定类型的值，也可以是 FileType 或字符串类型。例如 int 会将参数值转换为整数类型。
        :param 可选:[choices]: 该参数是一个可迭代对象，指定了参数值的可选范围。若命令行传入的参数值不在此范围内，就会报错。
        :param 必填:[required]: 这是一个布尔值，若设为 True，则表示该参数在命令行中是必需的，若未提供就会报错。
        :param 帮助:[help]: 该字符串用于提供该参数的帮助说明，当用户使用 --help 选项时，会显示此信息。
        :param 元变量名:[metavar]: 用于在帮助信息中代表参数值的名称。可以是一个字符串，也可以是一个字符串元组。
        :param 目标属性名:[dest]: 该参数指定在解析结果对象中存储该参数值的属性名。若未指定，会根据参数名称自动生成。
        :param 版本:[version]: 当 action 为 'version' 时，该参数指定要显示的版本信息。
        :param 其他关键字参数:[**kwargs]: 可以传入其他额外的关键字参数，以满足特定的自定义需求。
        :param action:
        :param nargs:
        :param const:
        :param default:
        :param type:
        :param choices:
        :param required:
        :param help:
        :param metavar:
        :param dest:
        :param version:
        :return:
        """
        if action or 动作:
            return super().add_argument(
                *名称或标志,
                action=动作 if 动作 is not None else action,
                # nargs=参数数量 if 参数数量 is not None else nargs,
                # const=常量值 if 常量值 is not None else const,
                # default=默认 if 默认 is not None else default,
                # type=类型 if 类型 is not None else type,
                # choices=可选 if 可选 is not None else choices,
                # required=必填 if 必填 is not None else required,
                help=帮助 if 帮助 is not None else help,
                # metavar=元变量名 if 元变量名 is not None else metavar,
                # dest=目标属性名 if 目标属性名 is not None else dest,
                # version=版本 if 版本 is not None else version,
                **其他关键字参数
            )
        return super().add_argument(
            *名称或标志,
            action=动作 if 动作 is not None else action,
            nargs=参数数量 if 参数数量 is not None else nargs,
            const=常量值 if 常量值 is not None else const,
            default=默认 if 默认 is not None else default,
            type=类型 if 类型 is not None else type,
            choices=可选 if 可选 is not None else choices,
            required=必填 if 必填 is not None else required,
            help=帮助 if 帮助 is not None else help,
            metavar=元变量名 if 元变量名 is not None else metavar,
            dest=目标属性名 if 目标属性名 is not None else dest,
            # version=版本 if 版本 is not None else version,
            **其他关键字参数
        )
    def print_help(self, file=None):
        # if self.parse_args().help:
        print(f"""
============================================= 帮助信息 =============================================
程序名称: {self.prog}
使用方法: {self.usage}
{self._actions[-1].help}
功能描述: {self.description}
使用示例: {self.epilog}
============================================= 帮助信息 =============================================
        """)

    def parse_args(self, args=None, namespace=None):
        arg = super().parse_args(args=args, namespace=namespace)
        try:
            if arg.help:
                self.print_help()
                sys.exit()
        except AttributeError:
            pass
        return arg

    def error(self, message: str):
        self.print_help()
        print(f"报错信息: {message}")
        sys.exit()


class ProteinGCN(GCNConv):
    """基本卷积核"""

    def edge_update(self) -> Tensor:
        return super().edge_update()

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]], out_channels: int,
            kernel_size=3, padding=1, relu: Any = nn.ReLU,
            **kwargs):
        super().__init__(
            in_channels=in_channels, out_channels=out_channels,
            **kwargs)
        self.kernel_size = kernel_size
        self.padding = padding
        self.relu = relu


class SuperParameters(object):
    device = DEVICE
    def __init__(
            self,
            label1="模型参数",
            save_model_path=None,
            model=None,
            model_name=None,
            conv=ProteinGCN,
            conv_name=None,
            in_channels=1280,
            hidden_channels=128,
            cls_hidden=1,
            out_channels=1,
            layer_number=3,

            bn=nn.BatchNorm1d,
            act_fun=nn.ReLU,
            fc=nn.Linear,
            global_pool=global_mean_pool,

            label2="数据参数",
            data_path=None,
            save_data_path=None,
            data_name=None,
            ratio=(0.70, 0.15, 0.15),
            feature_name=None,
            max_len=500,
            batch_size=1,
            seed=42,
            decimal=3,
            metrics_type='zh',
            show_report=False,
            save_model_dir=None,
            save_data_dir=None,

            label3="训练参数",
            epochs=2000,
            max_inv_epoch=2000,
            lr=0.01,
            step_size=50,
            gamma=0.95,
            criterion=torch.nn.MSELoss,
            optimizer=torch.optim.Adam,
            # optimizer=torch.optim.RMSprop,
            scheduler=torch.optim.lr_scheduler.StepLR,

            label4="其他参数",
            language='zh',
            report_dict=None,
            conv_dict=None,
            **kwargs,
    ):
        super(SuperParameters, self).__init__()
        # ======================================= 模型参数 =======================================
        self.label1 = label1
        """模型参数"""
        self.model = model if model is not None else None
        self.model_name: str = model_name if model_name is not None else "ProteinCNN"
        self.conv = conv if conv is not None else ProteinGCN
        self.conv_name: str = conv_name if conv_name is not None else "GCN"
        self.in_channels = in_channels if in_channels is not None else 1280
        self.hidden_channels = hidden_channels if hidden_channels is not None else 128
        self.cls_hidden = cls_hidden if cls_hidden is not None else 1
        self.out_channels = out_channels if out_channels is not None else 1
        self.layer_number = layer_number if layer_number is not None else 3
        self.bn = bn if bn is not None else nn.BatchNorm1d
        self.act_fun: nn.functional = act_fun if act_fun is not None else F.relu
        self.fc = fc if fc is not None else nn.Linear
        self.global_pool = global_pool if global_pool is not None else global_mean_pool
        # ======================================= 模型参数 =======================================

        # ======================================= 数据参数 =======================================
        self.label2 = label2
        """数据参数"""
        self.data_path = data_path if data_path is not None else ""
        self.data_path = data_path if data_path is not None else ""
        self.data_name: str = data_name if data_name is not None else "数据名称"
        self.ratio = ratio if ratio is not None else (0.70, 0.15, 0.15)
        self.feature_name: str = feature_name if feature_name is not None else "特征名称"
        self.max_len = max_len if max_len is not None else 500
        self.batch_size = batch_size if batch_size is not None else 1
        self.seed = seed if seed is not None else 42
        self.decimal = decimal if decimal is not None else 3
        self.save_model_dir: str = save_model_dir if save_model_dir is not None else os.path.join(DEFAULT_SAVE_DIR, 'models')
        self.save_data_dir: str = save_data_dir if save_data_dir is not None else os.path.join(DEFAULT_SAVE_DIR, 'data')
        self.save_model_path: str = save_model_path if save_model_path is not None else os.path.join(DEFAULT_SAVE_DIR, "model/PG16/PG16_ProteinCNN.pt")
        self.save_data_path: str = save_data_path if save_data_path is not None else os.path.join(DEFAULT_SAVE_DIR, "model/PG16/PG16_ProteinCNN.xlsx")
        # ======================================= 数据参数 =======================================

        # ======================================= 训练参数 =======================================
        self.label3 = label3
        """训练参数"""
        self.epochs = epochs if epochs is not None else 1000
        self.max_inv_epoch = max_inv_epoch if max_inv_epoch is not None else 300
        self.lr = lr if lr is not None else 0.001
        self.step_size = step_size if step_size is not None else 10
        self.gamma = gamma if gamma is not None else 0.95
        self.criterion = criterion if criterion is not None else torch.nn.MSELoss
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam
        self.scheduler = scheduler if scheduler is not None else torch.optim.lr_scheduler.StepLR
        # ======================================= 训练参数 =======================================

        # ======================================= 其他参数 =======================================
        self.label4 = label4
        """其他参数"""
        self.language = language if language is not None else "zh"
        self.show_report = show_report if show_report is not None else False
        self.report_dict = report_dict if report_dict is not None else ZH_REPORT_DICT
        self.report_df = report_dict if report_dict is not None else ZH_REPORT_DATA_FRAME
        self.conv_dict = conv_dict if conv_dict is not None else None
        # 更多其他参数
        for key, value in kwargs.items():
            setattr(self, key, value)
        # ======================================= 其他参数 =======================================

    def __str__(self):
        attr_str = "SuperParameters(\n"
        for attr, value in self.__dict__.items():
            attr_str += f"\t{attr}: {value}\n"
        attr_str += ")"
        return attr_str
# ================================================== 类定义 ==================================================
