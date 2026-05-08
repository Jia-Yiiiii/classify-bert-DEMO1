# classify-bert-DEMO1
基于BERT的今日头条新闻文本分类模型

---

## 📌 项目介绍
本项目使用 `bert-base-chinese` 预训练模型，对 **15 类中文新闻文本**进行自动分类，实现了从数据加载、模型训练、模型评估到单句预测的完整流程，并通过 SwanLab 可视化实验过程。

本项目在 3k 训练集上，最终在测试集上取得了 **83.65%** 的准确率，验证了BERT在中文文本分类任务上的有效性。

模型文件 best_model.pth 因体积较大未上传，可自行训练生成或通过提供的代码加载。
---

## 📊 模型效果可视化

### 1. 训练过程曲线
<p align="center">
  <img width="700" alt="训练过程曲线" src="https://github.com/user-attachments/assets/bc0cf990-76ab-40f1-adc8-45ef736d8b92">
</p>

### 2. 混淆矩阵热力图
<p align="center">
  <img width="700" alt="混淆矩阵" src="https://github.com/user-attachments/assets/eff7d3fe-6e6b-48e5-b33c-23f39d5d99b7">
</p>

### 3. 各类别准确率
<p align="center">
  <img width="700" alt="各类别准确率" src="https://github.com/user-attachments/assets/73966cdc-ae66-4e77-abab-704683b3f540">
</p>

### 4. 数据集标签分布
<p align="center">
  <img width="500" alt="标签分布" src="https://github.com/user-attachments/assets/a7c0cd06-e52c-4406-b2b3-12a1a4c7d9b0">
</p>

### 5. 模型预测置信度分布
<p align="center">
  <img width="700" alt="模型预测置信度分布" src="https://github.com/user-attachments/assets/40291400-a208-4127-ad98-bf54298cabe3">
</p>

### 6. 分类指标报告
<p align="center">
  <img width="800" alt="train_loss  train_acc" src="https://github.com/user-attachments/assets/43b80cd9-cb59-4e10-91f3-1e041ecf0688">
</p>
<p align="center">
  <img width="800" alt="dev_loss dev_acc" src="https://github.com/user-attachments/assets/303e0830-0767-4e99-acf9-85c46dd3971c">
</p>

<p align="center">
  <img width="450" alt="test_acc" src="https://github.com/user-attachments/assets/fc3d81e0-c037-4189-b438-3f55df429c40">
</p>
## 📉 模型表现分析
模型整体准确率 **83.65%**，表现优秀，但存在明显的短板类别，各类别表现差异较为突出：

### 表现最差的三个类别
- news_agriculture（农业新闻）
- news_finance（财经新闻）
- stock（股票）

### 短板类别指标偏低的主要原因
1.  样本数量极少，数据集存在严重的类别不平衡问题，模型缺乏足够的训练样本进行学习；
2.  三类文本的特征相似度较高，模型难以精准区分，容易出现互相误判的情况；
3.  样本不足导致模型学习不充分，对这类文本的泛化能力较弱，难以适应未见过的样本。

### 表现最好的三个类别
- news_car（汽车新闻）
- news_sports（体育新闻）
- news_edu（教育新闻）

这类类别的优势的在于：样本数量充足、文本特征清晰且具有高辨识度，模型能够充分学习其核心特征，因此准确率均达到 **88%~91%+**，分类效果稳定优秀。
---

## 📁 项目结构
```text
classify-bert-DEMO1/
├── DATA/                     # 数据集文件夹
│   ├── train_3k.txt          # 训练集
│   ├── dev_1k.txt            # 验证集
│   └── test_1k.txt           # 测试集
├── config/
│   └── Bert_Config_exp1.json # 模型超参数配置文件
├── main.py                   # 模型训练与测试主程序
├── predict.py                # 单句新闻分类预测代码
├── best_model.pth            # 训练好的最优模型权重文件
├── label_map.json            # 标签映射文件
├── requirements.txt          # 依赖库清单
└── README.md                 # 项目说明文档
