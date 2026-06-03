# classify-bert-DEMO1

基于 BERT 的今日头条新闻文本分类模型

---

## 项目介绍

本项目使用 `bert-base-chinese` 预训练模型，对 **15 类中文新闻文本**进行自动分类，实现了从数据加载、模型训练、模型评估到单句预测的完整流程，并通过 SwanLab 可视化实验过程。

模型在 3k 训练集上收敛稳定，最终在测试集上取得了 **83.4%** 的准确率，验证了 BERT 在中文文本分类任务上的有效性。

> 注：模型文件 `best_model.pth` 因体积较大未上传，可自行训练生成。

---

## 数据集

本项目使用今日头条新闻标题分类数据集。

- **数据来源**：今日头条客户端
- **采集时间**：2018 年 5 月
- **下载地址**：[toutiao-text-classfication-dataset](https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset)
- **分类数量**：15 类

### 数据划分

| 数据文件 | 样本数量 | 用途 |
| ------- | -------- | ---- |
| `train_3k.txt` | 3,000 | 训练集 |
| `dev_1k.txt` | 1,000 | 验证集 |
| `test_1k.txt` | 1,064 | 测试集 |

---

## 数据分析

<img width="1324" height="167" alt="数据样例" src="https://github.com/user-attachments/assets/705cac3e-63bb-4f29-8bd2-70ee0d8b7187" />

通过读取数据前五行发现，数据的构成为五个部分。预处理时，将每条新闻的**标题**和**关键词**用中文逗号拼接，作为模型的输入文本，以辅助模型进行分类。

---

## 模型指标

### 1.整体分类报告

<img width="648" height="589" alt="分类报告" src="https://github.com/user-attachments/assets/9a267544-402f-407a-b2e6-a08bff7c0555" />

###2.可视化分析
<img width="1118" height="867" alt="d3c00384075ec51186c35ead42415bee" src="https://github.com/user-attachments/assets/24de0152-bf3c-4320-bb89-95975d84cd5f" />

<img width="1499" height="611" alt="743010159ebbb6fde51a6b29ce4d78bd" src="https://github.com/user-attachments/assets/e7b37bd4-25ca-4474-8fb6-26d0572ed728" />

### 表现分析

#### 表现最好的三个类别

- `news_car`（汽车新闻）
- `news_sports`（体育新闻）
- `news_edu`（教育新闻）

#### 表现最差的三个类别

- `news_agriculture`（农业新闻）
- `news_finance`（财经新闻）
- `stock`（股票）

**原因分析**：

1. 样本数量极少，类别不平衡严重
2. 文本特征相似，易互相误判
3. 训练不足导致泛化能力较弱

---

## 单句预测演示

支持输入任意中文新闻文本，直接输出分类结果。

**示例**：

<img width="278" height="72" alt="预测示例" src="https://github.com/user-attachments/assets/2bec1acf-22da-4249-b539-6753a2f6ee9a" />



### 使用方式

运行 `Predict.py`，可自定义输入文本进行推理。

---

## 项目结构

```text
classify-bert-DEMO1/
├── DATA/                    # 数据集文件夹
│   ├── train_3k.txt         # 训练集
│   ├── dev_1k.txt           # 验证集
│   ├── test_1k.txt          # 测试集
│   └── label_map.json       # 标签映射文件
├── configs/                 # 配置文件目录
│   └── Bert_Config_exp1.json
├── model.py                 # 模型结构
├── Predict.py               # 测试/预测脚本
├── trainer.py               # 训练脚本
├── utils.py                 # 工具函数
├── requirements.txt         # 依赖
└── README.md                # 项目说明
