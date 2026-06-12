# classify-bert-DEMO1

基于 BERT 的今日头条新闻文本分类模型

---

## 项目介绍

本项目使用 bert-base-chinese 预训练模型，对 15 类中文新闻文本进行自动分类，实现了从数据加载、模型训练、模型评估到单句预测的完整流程，并通过 SwanLab 可视化实验过程。

模型在训练集上的损失函数持续下降，表明优化过程稳定、模型基本收敛。验证集准确率在第 2 轮达到最高 0.872，随后略有下降。通过早停机制设置patience=3在验证准确率连续 3 轮不再提升后停止训练，最终保存第 2 轮的模型。最终测试集准确率为 0.848。

> 注：模型文件 `best_model.pth` 因体积较大未上传。
> ## 模型保存

训练时通过 `savemodel()` 保存以下文件：
- `best_model.pth` - 模型权重
- `training_config.json` - 训练参数
- `DATA/label_id.txt` 和 `DATA/id_label.txt` - 标签映射
- `tokenizer/` - 分词

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
<img width="497" height="367" alt="image" src="https://github.com/user-attachments/assets/a3c7a756-7ed1-4b6a-8574-a290712d22fc" />
<img width="445" height="415" alt="image" src="https://github.com/user-attachments/assets/9c1e7257-d7a0-4d8d-a171-54c7ea2ffd61" />
<img width="482" height="334" alt="image" src="https://github.com/user-attachments/assets/7dc4644b-3792-4227-ab51-e40012a3aa3c" />
<img width="480" height="338" alt="image" src="https://github.com/user-attachments/assets/3f332929-2037-41dc-b666-a98ba7df2622" />
<img width="479" height="335" alt="image" src="https://github.com/user-attachments/assets/5fbeac08-21b6-4965-82cd-2818a8ee54e8" />

### 整体分类报告

<img width="636" height="615" alt="e765164df2e6cbf19ea1899768fa5f48" src="https://github.com/user-attachments/assets/2128dc01-6a45-425a-af3d-9932187999fa" />


### 可视化分析

<img width="1230" height="885" alt="846d94d1da156a82fdf81d9993054c1c" src="https://github.com/user-attachments/assets/a41e2863-93e1-43cc-9005-4b588d72f539" />
<img width="1488" height="610" alt="8cb36a522b23cb0eb6f4a7a24999b78f" src="https://github.com/user-attachments/assets/0a296d6a-6b07-4c35-b5ff-64090729537c" />

### 表现分析

模型整体表现良好，验证集最佳准确率达到87.2%，测试集准确率为84.8%。从各类别F1分数来看，汽车、教育、体育等类别表现优秀，特征明显易于区分；故事和股票（表现最差。从混淆矩阵分析，主要问题集中在财经与股票相互混淆，故事与文化娱乐类边界模糊总体而言，受限于训练数据量，部分语义相近的类别区分困难，后续可通过补充标注数据或数据增强来进一步优化。

## 单句预测演示

支持输入任意中文新闻文本，直接输出分类结果。

<img width="330" height="79" alt="image" src="https://github.com/user-attachments/assets/f669e63f-f75f-4f9c-bd3d-7f7c0d7e4975" />

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
├── Predict.py               # 预测脚本
├── trainer.py               # 训练脚本
├── utils.py                 # 工具函数
├── requirements.txt         # 依赖
└── README.md                # 项目说明
