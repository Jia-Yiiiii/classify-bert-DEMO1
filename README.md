# classify-bert-DEMO1
基于 BERT 的今日头条新闻文本分类模型

---

## 📌 项目介绍
本项目使用 `bert-base-chinese` 预训练模型，对 **15 类中文新闻文本**进行自动分类，实现了从数据加载、模型训练、模型评估到单句预测的完整流程，并通过 SwanLab 可视化实验过程。

本项目在 3k 训练集上，最终在测试集上取得了 **83.65%** 的准确率，验证了 BERT 在中文文本分类任务上的有效性。

模型文件 `best_model.pth` 因体积较大未上传，可自行训练生成。

---

## 📑 目录
- [📌 项目介绍](#-项目介绍)
- [📊 模型效果可视化](#-模型效果可视化)
- [📁 项目结构](#-项目结构)

---

## 📊 模型效果可视化

### 1. 训练过程曲线
![训练过程曲线](https://github.com/user-attachments/assets/bc0cf990-76ab-40f1-adc8-45ef736d8b92)

### 2. 混淆矩阵
![混淆矩阵](https://github.com/user-attachments/assets/eff7d3fe-6e6b-48e5-b33c-23f39d5d99b7)

### 3. 各类别准确率
![各类别准确率](https://github.com/user-attachments/assets/73966cdc-ae66-4e77-abab-704683b3f540)

### 4. 标签分布
![标签分布](https://github.com/user-attachments/assets/a7c0cd06-e52c-4406-b2b3-12a1a4c7d9b0)

### 5. 模型预测置信度分布
![模型预测置信度分布](https://github.com/user-attachments/assets/40291400-a208-4127-ad98-bf54298cabe3)

### 6. SwanLab 可视化
![SwanLab1](https://github.com/user-attachments/assets/0f4fcc65-880c-4569-86f1-93b55586cdb1)
![SwanLab2](https://github.com/user-attachments/assets/91e4d574-cc48-4eed-9b61-feed9964c722)
![SwanLab3](https://github.com/user-attachments/assets/09266721-7895-43cd-9d55-67ae9d1dd095)

---

## 模型表现分析
模型整体准确率 **83.65%**，表现优秀，但存在明显的短板类别，各类别表现差异较为突出。

### 表现最差的三个类别
- news_agriculture（农业新闻）
- news_finance（财经新闻）
- stock（股票）

### 指标偏低的主要原因
1. 样本数量极少，数据集存在严重的类别不平衡问题，模型缺乏足够的训练样本进行学习；
2. 三类文本特征相似度较高，模型难以精准区分，容易出现互相误判；
3. 样本不足导致模型学习不充分，泛化能力较弱。

### 表现最好的三个类别
- news_car（汽车新闻）
- news_sports（体育新闻）
- news_edu（教育新闻）

这类类别样本充足、文本特征清晰，模型能够充分学习核心特征，准确率均达到 **88%~91%+**，分类效果稳定优秀。

---

## 📁 项目结构
```text
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
