# -*- coding: utf-8 -*-

"""
----
文件名称: __init__.py
----
\\
----
模块概述: 特征设计
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
    - math
    - os
    - numpy as np
    - torch
    - Bio.PDB.PDBParser as PDBParser
\\
\\
----
使用示例：
----
    - feature.P_features(sequences='TEST', max_len=4, ph=11.5)
    - feature.onehot_features(sequences='TEST', max_len=4)
    - feature.esm2_features(sequences='TEST', max_len=4)
    - feature.sequences_features('../exclude/test.pdb')
\\
\\
----    
异常处理：
----
    - Exception
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
    'P_features',
    'onehot_features',
    'esm2_features',
    'edge_features',
    'sequences_features',
]
__author__ = '陆家立'
__email__ = '996153075@qq.com'
__version__ = '1.0.0'

import math
import os

import numpy as np
import torch
from Bio.PDB import PDBParser

from evolution import template
from evolution.esm.pretrained import load_model_and_alphabet_hub
from evolution.models.read import get_esm2_tokenizer
from evolution.template import AMINO_ACIDS, DEVICE

# ================================================== 特殊属性与导入 ==================================================


# ================================================== 全局变量 ==================================================
tokenizer = get_esm2_tokenizer()
# ================================================== 全局变量 ==================================================


# ================================================== 函数定义 ==================================================
def P_features(sequences, max_len, ph: float = 11.5) -> np.ndarray:
    """
    蛋白序列PC特征

    :example:
    >>> Result = P_features(sequences="TEST", max_len=4, ph=7)
    >>> isinstance(Result, np.ndarray)
    True

    :param sequences: 蛋白质序列
    :param max_len: 最大序列长度
    :param ph: 给定pH值
    :return: 蛋白质特征
    """
    import Bio
    from Bio.SeqUtils import ProtParam
    protein_analysis = ProtParam.ProteinAnalysis(sequences)
    properties = {
        '芳香性': protein_analysis.aromaticity(),
        '不稳定指数': protein_analysis.instability_index(),
        "G+C百分比": Bio.SeqUtils.gc_fraction(sequences),
        "G+C总含量": Bio.SeqUtils.GC123(sequences)[0],
        '水病总平均数': protein_analysis.gravy(),
        '等电点': protein_analysis.isoelectric_point(),
        '分子量': Bio.SeqUtils.molecular_weight(sequences, seq_type="protein"),
        '螺旋、车削和片材的分数': list(protein_analysis.secondary_structure_fraction()),
        "GC偏斜": Bio.SeqUtils.GC_skew(sequences),
        '给定PH环境下的电荷': protein_analysis.charge_at_pH(pH=ph),
        '摩尔消光系数': list(protein_analysis.molar_extinction_coefficient())
    }
    properties_list = []
    for key, value in properties.items():
        # print(key, value)
        if isinstance(value, int) or isinstance(value, float): properties_list.append(value)
        if isinstance(value, list): properties_list.extend(value)
    result = np.array(properties_list)
    # result = np.pad(result, ((0, max_len - result.shape[0]), (0, 0)), 'constant', constant_values=0)
    return result


def onehot_features(sequences, max_len) -> np.ndarray:
    """
    蛋白序列独热编码特征

    :example:
    >>> Result = onehot_features(sequences="TEST", max_len=4)
    >>> isinstance(Result, np.ndarray)
    True

    :param sequences: 蛋白序列
    :param max_len: 最大序列长度
    :return: 蛋白 onehot 矩阵
    """
    aa_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    one_hot_list = []
    for seq in sequences:
        zero_vector = [0] * 20
        if seq in AMINO_ACIDS:
            zero_vector[aa_idx[seq]] = 1
            one_hot_list.append(zero_vector)
    result = np.array(one_hot_list)
    result = np.pad(result, ((0, max_len - result.shape[0]), (0, 0)), 'constant', constant_values=0)
    return result


def esm2_features(sequences, max_len):
    """
    蛋白序列ESM2特征2

    :example:
    >>> Result = esm2_features(sequences="TEST", max_len=4)
    >>> isinstance(Result, np.ndarray)
    True

    :param sequences: 蛋白序列
    :param max_len: 最大序列长度
    :return:
    """
    esm2_model, alphabet = load_model_and_alphabet_hub("esm2_t33_650M_UR50D")
    esm2_model.to(DEVICE)
    esm2_model.eval()
    try:
        batch_converter = alphabet.get_batch_converter()
        _, _, batch_tokens = batch_converter([("protein", sequences)])   # 转换序列为tokens
        batch_tokens = batch_tokens.to(DEVICE)
        with torch.no_grad():
            if DEVICE.type == 'cuda':
                with torch.amp.autocast('cuda', dtype=torch.float32):
                    results = esm2_model(batch_tokens, repr_layers=[10], return_contacts=False)
            else:
                results = esm2_model(batch_tokens, repr_layers=[10], return_contacts=False)
            result = results['representations'][10].to('cpu')[0][1:-1].numpy()
            result = np.pad(result, ((0, max_len - result.shape[0]), (0, 0)), 'constant', constant_values=0)
            return result
    except Exception as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise print(f"忽略的异常: {e}")


def edge_features(pdb_path):
    """
    边特征

    :example:
    >>> Result = edge_features('../exclude/4_2_J45B.pdb')
    >>> isinstance(Result, list)
    True

    :param pdb_path: 蛋白 pdb 文件
    :return:
    """
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB 文件错误: {pdb_path} 不存在")
    # 创建 PDBParser 对象
    parser = PDBParser(QUIET=True)
    try:
        # 解析 PDB 文件
        structure = parser.get_structure('protein', pdb_path)
    except Exception as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise print(f"忽略的异常: {e}")
        else:
            raise print(f"PDB 文件错误: {pdb_path} 解析错误")
    ca_coordinates = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_atom = residue['CA']
                    ca_coordinates.append(ca_atom.coord.tolist())
    # 计算邻居矩阵
    key_dict = {}
    for index, items in enumerate(ca_coordinates):
        x1, y1, z1 = items
        key_dict[index] = []
        for idx, item in enumerate(ca_coordinates):
            if index == idx:
                key_dict[index].append(0)
                continue
            x2, y2, z2 = item
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
            key_dict[index].append(distance)
    # 获取共边矩阵
    one_edge = []
    two_edge = []
    for key1, items1 in key_dict.items():
        key1_list = []
        key2_list = []
        for key2, item in enumerate(items1):
            if item and item < 8:
                key1_list.append(key1)
                key2_list.append(key2)
        one_edge.extend(key1_list)
        two_edge.extend(key2_list)
    return [one_edge, two_edge]


def sequences_features(pdb_path):
    """
    获取蛋白 PDB 氨基酸序列

    :example:
    >>> Result = sequences_features('../exclude/test.pdb')
    >>> isinstance(Result, tuple)
    True

    :param pdb_path: 蛋白 PDB 文件
    :return: 蛋白 sequences 序列
    :raise FileNotFoundError: 文件错误
    """
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"文件错误: 获取蛋白 PDB 氨基酸序列失败 {pdb_path} 不存在")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)

    amino_acid_sequence = ''
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == ' ':  # Check if the residue is a standard amino acid
                    res_name = residue.get_resname()
                    if res_name in template.AMINO_ACIDS_DICT:
                        amino_acid_sequence += template.AMINO_ACIDS_DICT[res_name]

    return amino_acid_sequence, len(amino_acid_sequence)
# ================================================== 函数定义 ==================================================


# ================================================== 类定义 ==================================================

# ================================================== 类定义 ==================================================
