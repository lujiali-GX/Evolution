import numpy as np
import pandas as pd
from torch_geometric.data import Data

from evolution import feature, template, compute
from evolution.data import preconditioning
from evolution.compute import compute_features, compute_torch_features
from evolution.data.preconditioning import processing_file_name


# ================================================== 特殊属性与导入 ==================================================


# ================================================== 全局变量 ==================================================

# ================================================== 全局变量 ==================================================


# ================================================== 函数定义 ==================================================
def test_compute_features():
    cf = feature.P_features
    result = compute.compute_features(sequences='TEST', cf=cf, max_len=5)
    assert isinstance(result, np.ndarray)


def test_compute_Data():
    pdbPath = template.TEMP_DIR+'/PG16.pdb'
    print(pdbPath)
    Sequences, _ = feature.sequences_features(pdbPath)
    result = compute.compute_Data(
        cf = feature.esm2_features,
        cf_name = 'ESM2',
        idx = 0,
        name = 'test',
        pdb_path = pdbPath,
        sequences = Sequences,
        max_len = len(Sequences),
        y = 1.1,)
    isinstance(result, Data)


def test_compute_torch_features():
    import json
    allDataFrame = json.load(open(template.TEMP_DIR+'/all.json'))
    newCf = feature.esm2_features
    for key, value in allDataFrame.items():
        if key != 'PG16': continue
        newDf = pd.DataFrame(dict(value), columns=value.keys())
        maxLen = max(newDf['Len'])
        result = compute_torch_features(df=newDf.iloc[:3], cf=newCf, max_len=maxLen,
            df_name='PG16', cf_name='ESM2',
            sf=template.TEMP_DIR+'/test.pt',
            is_save=True, cache=True)
        assert isinstance(result, bool)
# ================================================== 函数定义 ==================================================
