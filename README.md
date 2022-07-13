## 面向黑灰产的恶意变体字识别

#### 1. 项目目录

```markdown
|-- Code												// 代码运行文件夹
|		|-- soundshapecode								// 所需第三方module
|		|-- Utils											
|		|		|-- __init__.py					
|		|		|-- config.py							// 项目配置
|		|		|-- Dictionary.py						// 字典对象，word--index
|		|		|-- Variant_word.py						// 变体字 Dataset 对象
|		|		|-- VariantNeedleman_Wunsch.py			// 文本相似度对齐
|		|-- baseline.ipynb								// 基线文件
|		|-- preprocess.ipynb
|		|-- test.ipynb
|		|-- train.ipynb									// 未定
|-- Data
|		|-- Dataset
|		|		|-- data.csv
|		|		|-- train_data.csv
|		|		|-- test_data.csv
|		|-- source_vocal.pkl							// 源词表
|		|-- target_vocal.pkl							// 目标词表
|-- Logs
|-- Model
|		|-- __init__.py
|		|-- BaselineModel.py							// 基线模型系统
|		|-- CustomScheduleLearningRate.py
|		|-- MyTransformer_parts.py							
|		|-- transformer_parts.py
|		|-- TranslationModel.py							// 翻译模型
|-- Weights
Requirements.txt														
.gitignore
setup.sh												// 安装所需 module
```
----

