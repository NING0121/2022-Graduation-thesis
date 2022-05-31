## 面向黑灰产的恶意变体字识别

#### 1. 项目目录

```markdown
|-- Code											// ipynb代码运行文件夹
|	|-- baseline.ipynb
|	|-- preprocess.ipynb
|	|-- test.ipynb
|-- Data
|	|-- Dataset
|	|	|-- data.csv
|	|	|-- train_data.csv
|	|	|-- test_data.csv
|	|-- source_vocal.pkl							// 源词表
|	|-- target_vocal.pkl							// 目标词表
|	|-- src_idx2tgt_idx.pkl							// 映射词表
|-- Model											// 存放pl训练网络
|	|-- __init__.py
|	|-- ConvS2SModel.py
|	|-- RNNSearchModel.py
|	|-- TransformerModel.py
|-- Networks										// 存放模型网络
|	|-- __init__.py
|	|-- ConvS2S_parts.py
|	|-- ConvS2S.py
|	|-- MyTransformer_parts.py
|	|-- RNNSearch.py
|	|-- TranslationModel.py
|-- Logs											// 用于存放训练Logs
|-- Weighs											// 存放checkpoints
|-- soundshapecode									// 所需第三方module
train.py											// 训练文件
config.py											// 配置文件
data_loader.py										// 数据构建文件
Requirements.txt														
.gitignore
setup.sh											// 安装所需 module
test_run.out										// 训练输出
```

#### 2.使用方法

1. 运行 setup.sh 完成相关环境配置；
2. 使用 train.py 完成模型训练；相关使用方法已经写在py文件中。