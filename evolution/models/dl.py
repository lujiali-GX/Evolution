# -*- coding: utf-8 -*-

"""
----
文件名称: dl.py
----
\\
----
模块概述: 深度学习模型
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
    - dl.main()
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
]
__author__ = '陆家立'
__email__ = '996153075@qq.com'
__version__ = '1.0.0'

from typing import Union, Tuple, Any

import torch
from torch import nn, Tensor
from torch.nn import Conv3d
from torch_geometric.nn import global_mean_pool, TAGConv, ClusterGCNConv, FiLMConv, SAGEConv, TransformerConv, MFConv, \
    GATConv, GCNConv, GINConv, GraphConv, ChebConv, ARMAConv, SGConv, APPNP
import torch.nn.functional as F
from torch_geometric.typing import Adj

from evolution.template import SuperParameters


# ================================================== 特殊属性与导入 ==================================================


# ================================================== 全局变量 ==================================================

# ================================================== 全局变量 ==================================================


# ================================================== 函数定义 ==================================================

# ================================================== 函数定义 ==================================================


# ================================================== 类定义 ==================================================
class ProteinTAGConv(TAGConv):
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


class ProteinClusterGCN(ClusterGCNConv):
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


class ProteinFiLM(FiLMConv):

    def edge_update(self, *args, **kwargs) -> Tensor:
        # 调用父类 FiLMConv 的 edge_update 方法
        return super().edge_update()

    def message_and_aggregate(self, edge_index: Adj) -> Tensor:
        return super().message_and_aggregate(edge_index=edge_index)

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


class ProteinSAGE(SAGEConv):
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


class ProteinTransformer(TransformerConv):

    def edge_update(self) -> Tensor:
        return super().edge_update()

    def message_and_aggregate(self, edge_index: Adj) -> Tensor:
        return super().message_and_aggregate(edge_index=edge_index)

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


class ProteinMF(MFConv):

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


class ProteinGAT(GATConv):

    def message_and_aggregate(self, edge_index: Adj) -> Tensor:
        return super().message_and_aggregate(edge_index=edge_index)

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


class ProteinGIN(GINConv):

    def edge_update(self) -> Tensor:
        return super().edge_update()

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]], out_channels: int,
            kernel_size=3, padding=1, relu: Any = nn.ReLU,
            **kwargs):
        gin_nn = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_channels, out_features=out_channels),
            relu(),
            torch.nn.Linear(in_features=out_channels, out_features=out_channels)
        )
        super().__init__(nn=gin_nn, **kwargs)
        self.kernel_size = kernel_size
        self.padding = padding
        self.relu = relu


class ProteinGraph(GraphConv):
    """通用卷积核"""

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


class ProteinCheb(ChebConv):
    """切比雪夫多项式卷积核"""

    def message_and_aggregate(self, edge_index: Adj) -> Tensor:
        return super().message_and_aggregate(edge_index=edge_index)

    def edge_update(self) -> Tensor:
        return super().edge_update()

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]], out_channels: int,
            kernel_size=3, padding=1, relu: Any = nn.ReLU,
            **kwargs):
        super().__init__(
            in_channels=in_channels, out_channels=out_channels,
            K=kernel_size,
            **kwargs)
        self.kernel_size = kernel_size
        self.padding = padding
        self.relu = relu


class ProteinARMA(ARMAConv):
    """自回归移动平均卷积核"""

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


class ProteinSG(SGConv):
    """自回归移动平均卷积核"""

    def edge_update(self) -> Tensor:
        return super().edge_update()

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]], out_channels: int,
            kernel_size=3, padding=1, relu: Any = nn.ReLU,
            **kwargs):
        super().__init__(
            in_channels=in_channels, out_channels=out_channels,
            K=kernel_size,
            **kwargs)
        self.kernel_size = kernel_size
        self.padding = padding
        self.relu = relu


class ProteinAPPNP(APPNP):
    def edge_update(self) -> Tensor:
        return super().edge_update()

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]], out_channels: int,
            kernel_size=3, padding=1, relu: Any = nn.ReLU,
            **kwargs):
        super(ProteinAPPNP, self).__init__(K=kernel_size, alpha=kernel_size*0.1)
        GCNConv(in_channels, out_channels, **kwargs)
        self.kernel_size = kernel_size
        self.padding = padding
        self.relu = relu


class ProteinC3D(Conv3d):
    """传统卷积核"""
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]], out_channels: int,
            kernel_size=3, padding=1, relu: Any = nn.ReLU,
            **kwargs):
        super().__init__(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, padding=padding,
            **kwargs)
        self.kernel_size = kernel_size
        self.padding = padding
        self.relu = relu

DL_CONV_DICT = {
    "TAG": ProteinTAGConv,
    "ClusterGCN": ProteinClusterGCN,
    "FiLM": ProteinFiLM,
    "SAGE": ProteinSAGE,
    "Transformer": ProteinTransformer,
    "MF": ProteinMF,
    "GAT": ProteinGAT,
    "GCN": ProteinGCN,
    "GIN": ProteinGIN,
    "Graph": ProteinGraph,
    "Cheb": ProteinCheb,
    "ARMA": ProteinARMA,
    "SG": ProteinSG,
    "APPNP": ProteinAPPNP,
    "Conv3d": ProteinC3D,
}
"""深度学习模型字典"""



class EarlyStopping:
    def __init__(self, patience=300, verbose=False, delta=0, save_path=None):
        """
        :param patience: 当验证集性能在多少个 epoch 内没有提升时停止训练
        :param verbose: 是否打印早停信息
        :param delta: 衡量验证集性能提升的最小变化量
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        当验证集损失值降低时保存模型
        """
        if self.verbose:
            print(f'验证损失减少 ({self.val_loss_min:.6f} --> {val_loss:.6f}).  保存模型 ...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss


class ProteinCNN(nn.Module):
    def __init__(self, super_parameters: SuperParameters):
        # 调用父类 nn.Module 的构造函数
        super(ProteinCNN, self).__init__()
        in_channels = super_parameters.in_channels
        hidden_channels = super_parameters.hidden_channels
        cls_hidden = super_parameters.cls_hidden
        out_channels = super_parameters.out_channels

        self.batch_size = super_parameters.batch_size
        self.layer_number = super_parameters.layer_number
        self.relu = super_parameters.act_fun
        self.global_pool = super_parameters.global_pool

        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()

        for i in range(self.layer_number):
            if i == 0: cov = super_parameters.conv(in_channels, hidden_channels)
            else: cov = super_parameters.conv(hidden_channels, hidden_channels)
            self.conv.append(cov)
            self.bn.append(super_parameters.bn(hidden_channels))

        self.linear1 = nn.Linear(hidden_channels, cls_hidden)
        self.linear2 = nn.Linear(cls_hidden, out_channels)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.LeakyReLU()  # 使用 LeakyReLU 避免死区现象
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(self.layer_number):
            x = F.relu(self.bn[i](self.conv[i](x, edge_index)))
            # x = self.relu(self.bn[i](self.conv[i](x, edge_index)))

        x = self.global_pool(x, batch)
        x = self.linear1(x)
        x = self.linear2(x)
        return self.fc(x)


class ALLModel(nn.Module):
    def __init__(self, super_parameters: SuperParameters):
        super(ALLModel, self).__init__()

        traits_number = super_parameters.in_channels
        conv_hidden = super_parameters.hidden_channels
        cls_hidden = super_parameters.cls_hidden
        # hidden_size = super_parameters.hidden_size

        self.layer_number = super_parameters.layer_number
        self.device = super_parameters.device
        self.conv = nn.ModuleList()
        # if super_parameters.model_name != 'GIN':
        for i in range(self.layer_number):
            if i == 0:
                self.conv.append(super_parameters.conv(traits_number, conv_hidden))
            else:
                self.conv.append(super_parameters.conv(conv_hidden, conv_hidden))

        self.conv = self.conv.to(self.device)

        self.linear1 = nn.Linear(conv_hidden, cls_hidden)
        self.linear2 = nn.Linear(cls_hidden, 1)
        self.fc = nn.Sequential(
            nn.Linear(1, 1),
            nn.LeakyReLU()  # 使用 LeakyReLU 避免死区现象
        )

    def forward(self, mol):
        mol = mol.to(self.device)
        x = mol.x.to(self.device)
        edge_index = mol.edge_index.to(self.device)
        res = x
        for i in range(self.layer_number):
            res = self.conv[i](res, edge_index)

        # 使用全局平均池化聚合节点特征
        res = global_mean_pool(res, mol.batch)

        res = self.linear1(res)
        res = self.linear2(res)
        return self.fc(res)
# ================================================== 类定义 ==================================================
