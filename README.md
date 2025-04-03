# 基于 ESM2 生物蛋白特征的IC50预测模型
- **Evolution**:
- **简介**: 本项目实现从蛋白质PDB结构到IC50的预测模型。

----

# 目录
  - [安装](#安装)
  - [使用方法](#使用方法)
  - [项目结构](#项目结构)
  - [配置](#配置)
  - [许可证](#许可证)
  - [联系与感谢](#联系与感谢)
  
----

# 安装
- **环境**:
  - 系统：Linux (Ubuntu 24.04)
  - 编程：python >= 3.12
- **安装步骤**：
```bash
# 克隆项目仓库
git clone https://github.com/lujiali-GX/EVolution.git

# 进入项目目录
cd EVolution
```

```bash
# 创建项目环境
conda env create -f environment.yml
conda activate evolution
```
或者
```bash
# 创建项目环境
conda create -n evolution python=3.12
conda activate evolution

# 安装依赖包
pip install -r requirements.txt
```

----
# 使用方法
- **基本使用**:
```bash
python3 data_preprocess.py -i 原始XLSX文件目录 -p 原始PDB文件目录 -op 输出PDB目录 -oj 输出JSON文件目录 -od 输出原始数据集目录 -oas 输出数据集目录 -omd 输出机器学习数据集目录 -ods 输出深度学习数据集目录
```
```bash
python3 ml_prediction_report.py -omd 输入机器学习模型数据集目录 -omm 输出机器学习模型目录 -omd 输出机器模型预测报告目录 -l 报告语言
```
```bash
python dl_prediction_report.py -odd 输入深度学习模型数据集目录 -odr 输出深度学习模型预测报告目录 -d 设备 -l 报告语言
```
- **示例**:
```bash
python3 data_preprocess.py -i ./XLSX -p ./RAW_PDB -op ./PDB -oj ./JSON -od ./DATA -oas ./ALL_DATASET -omd ./ML_DATASET  -ods ./DATASET
```
```bash
python3 ml_prediction_report.py -omd ./ML_DATASET -omm ./ML_MODEL -omd ./REPORT/ML -l zh
```
```bash
python dl_prediction_report.py -odd ./DL_DATASET -odr ./REPORT/DL -d cuda -l zh
```
----

# 项目结构
```
Evolution/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ environment.yml
├─ evolution.yml
├─ evolution/
│  ├─ compute
│  ├─ data
│  ├─ esm
│  ├─ feature
│  ├─ models
│  ├─ template
│  ├─ __init__.py
│  ├─ data_processing.py
│  ├─ dl_prediction_report.py
│  ├─ ml_prediction_report.py
│  └─ ersion.py
├─ temp/
│  ├─ data
│  ├─ all.json
│  └─ test.pt
└─ tests/
   ├─ __init__.py
   ├─ test_compute.py
   ├─ test_data.py
   ├─ test_dl.py
   ├─ test_feature.py
   └─ test_ml.py
```

----

# 配置
- **依赖库**:
  - python >= 3.12
  - torch~=2.6.0+cu126
  - torchvision
  - torchaudio
  - cloud-tpu-client
  - datasets~=3.4.1
  - transformers~=4.48.1
  - evaluate~=0.4.3
  - biopython~=1.85
  - pandas~=2.2.3
  - numpy~=1.26.4
  - scikit-learn~=1.6.1
  - xgboost~=3.0.0
  - scipy~=1.15.2
  - einops~=0.8.1
  - biotite~=0.41.2
  - requests~=2.32.3
  - huggingface_hub~=0.29.3
  - openpyxl
  - sentencepiece
  - torch-geometric~=2.6.1
  - tqdm~=4.67.1
- **模板文件[evolution.template]**
  - **ESM2预训练模型目录**: template.ESM2_DIR
  - **本地默认保存目录**: template.DEFAULT_SAVE_DIR
  - **原始XLSX数据目录**: template.RAW_DATA_DIR
  - **原始PDB数据目录**: template.RAW_PDB_DIR
----

----

### 许可证
本项目采用 [MIT许可证](LICENSE)

----

### 联系与感谢
- **联系方式**: 996153075@qq.com
- **致谢**: 感谢 [ESM2预训练模型](https://github.com/facebookresearch/esm) 和 [BioPython](https://github.com/biopython/biopython) 等开源项目。
