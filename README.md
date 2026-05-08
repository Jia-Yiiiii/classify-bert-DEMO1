# classify-bert-DEMO1
基于BERT的今日头条新闻文本分类模型

---

## 📌 项目介绍
本项目使用 `bert-base-chinese` 预训练模型，对 **15 类中文新闻文本**进行自动分类，实现了从数据加载、模型训练、模型评估到单句预测的完整流程，并通过 SwanLab 可视化实验过程。

本项目在 3k 训练集上，最终在测试集上取得了 **83.65%** 的准确率，验证了BERT在中文文本分类任务上的有效性。

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
