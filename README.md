# classify-bert-DEMO1
基于 BERT 的今日头条新闻文本分类模型

---

## 📌 项目介绍
本项目使用 `bert-base-chinese` 预训练模型，对 **15 类中文新闻文本**进行自动分类，实现了从数据加载、模型训练、模型评估到单句预测的完整流程，并通过 SwanLab 可视化实验过程。

本项目在 3k 训练集上，最终在测试集上取得了 **83.65%** 的准确率，验证了BERT在中文文本分类任务上的有效性。

---

## 📑 目录
- [📌 项目介绍](#-项目介绍)
- [📊 模型效果可视化](#-模型效果可视化)
  - [1. 训练过程曲线](#1-训练过程曲线)
  - [2. 混淆矩阵](#2-混淆矩阵)
  - [3. 各类别准确率](#3-各类别准确率)
  - [4. 标签分布](#4-标签分布)
  - [5. 模型预测置信度分布](#5-模型预测置信度分布)
- [📁 项目结构](#-项目结构)
- [🚀 快速开始](#-快速开始)
  - [1. 环境依赖](#1-环境依赖)
  - [2. 数据准备](#2-数据准备)
  - [3. 训练模型](#3-训练模型)
  - [4. 模型预测](#4-模型预测)
- [📄 许可证](#-许可证)

---

## 📊 模型效果可视化
### 1. 训练过程曲线
训练过程中，模型在训练集和验证集上的损失与准确率变化如下图所示：
<p align="center">
  <img width="700" alt="训练过程曲线" src="https://github.com/user-attachments/assets/bc0cf990-76ab-40f1-adc8-45ef736d8b92">
</p>

### 2. 混淆矩阵
混淆矩阵展示了模型在各类别上的预测情况，对角线上的值为预测正确的样本数：
<p align="center">
  <img width="700" alt="混淆矩阵" src="https://github.com/user-attachments/assets/eff7d3fe-6e6b-48e5-b33c-23f39d5d99b7">
</p>

### 3. 各类别准确率
每个类别的单独准确率表现：
<p align="center">
  <img width="700" alt="各类别准确率" src="https://github.com/user-attachments/assets/73966cdc-ae66-4e77-abab-704683b3f540">
</p>

### 4. 标签分布
数据集的类别样本数量分布：
<p align="center">
  <img width="500" alt="标签分布" src="https://github.com/user-attachments/assets/a7c0cd06-e52c-4406-b2b3-12a1a4c7d9b0">
</p>

### 5. 模型预测置信度分布
模型对预测结果的置信度分布情况：
<p align="center">
  <img width="700" alt="模型预测置信度分布" src="https://github.com/user-attachments/assets/40291400-a208-4127-ad98-bf54298cabe3">
</p>

---

## 📁 项目结构
```tree
classify-bert-DEMO1/
├── DATA/                     # 数据集文件夹
│   ├── .gitkeep              # 占位文件
│   ├── train_3k.txt          # 训练集
│   ├── dev_1k.txt            # 验证集
│   └── test_1k.txt           # 测试集
├── config/                   # 配置文件目录
│   ├── .gitkeep              # 占位文件
│   ├── Bert_Config_exp1.json # 模型超参数配置
│   └── label_map.json        # 标签映射文件
├── Model.py                  # 模型训练与测试主程序
├── Predict.py                # 单句新闻分类预测
└── README.md                 # 项目说明文档
