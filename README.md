# classify-bert-DEMO1
基于BERT的今日头条新闻文本分类模型

---

## 📌 项目介绍
本项目使用 `bert-base-chinese` 预训练模型，对 **15 类中文新闻文本**进行自动分类，实现了从数据加载、模型训练、模型评估到单句预测的完整流程，并通过 SwanLab 可视化实验过程。

本项目在 3k 训练集上，最终在测试集上取得了 **83.65%** 的准确率，验证了BERT在中文文本分类任务上的有效性。
<img width="600" height="500" alt="3918401f1e0fc0c4522796b7cae81338" src="https://github.com/user-attachments/assets/bc0cf990-76ab-40f1-adc8-45ef736d8b92" />
<img width="1763" height="947" alt="8ed7247f2dc7a05b24096d9d5bf51acf" src="https://github.com/user-attachments/assets/eff7d3fe-6e6b-48e5-b33c-23f39d5d99b7" />
<img width="1476" height="616" alt="57a54160eb4aca4f47100fa70dcd28c7" src="https://github.com/user-attachments/assets/73966cdc-ae66-4e77-abab-704683b3f540" />
<img width="967" height="925" alt="3ba5c9c2db08bb8e16099c10fdcf6125" src="https://github.com/user-attachments/assets/a7c0cd06-e52c-4406-b2b3-12a1a4c7d9b0" />
<img width="1238" height="600" alt="ed3d0478fb7aff022afd2febae53b7e9" src="https://github.com/user-attachments/assets/40291400-a208-4127-ad98-bf54298cabe3" />

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
