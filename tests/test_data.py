# ================================================== 函数定义 ==================================================
import os

import numpy as np

from evolution import data, template
from evolution.data import preconditioning


def test_add_noise():
    result = data.add_noise(np.array([0, 1, 2, 3, 4]), 0.01)
    assert isinstance(result, np.ndarray)

def test_processing_file_dir():
    input_dir = template.RAW_DATA_DIR
    pdb_dir = template.PDB_DIR
    result = preconditioning.processing_file_dir(
        input_dir=input_dir,
        pdb_dir=pdb_dir,
        p_type='dict'
    )
    assert isinstance(result, dict)

def test_save_json_data():
    input_dir = template.RAW_DATA_DIR
    pdb_dir = template.PDB_DIR
    out_path = os.path.join(template.TEMP_DIR, "all.json")
    result_dict = preconditioning.processing_file_dir(input_dir, pdb_dir)
    assert isinstance(result_dict, dict)
    result = data.save_json_data(data=result_dict, path=out_path)
    assert isinstance(result, bool)

    return out_path
# ================================================== 函数定义 ==================================================


# ================================================== 类定义 ==================================================

# ================================================== 类定义 ==================================================