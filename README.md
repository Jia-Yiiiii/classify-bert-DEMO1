# classify-bert-DEMO1
基于 BERT 的今日头条新闻文本分类模型

---

## 项目介绍
本项目使用 `bert-base-chinese` 预训练模型，对 **15 类中文新闻文本**进行自动分类，实现了从数据加载、模型训练、模型评估到单句预测的完整流程，并通过 SwanLab 可视化实验过程。

模型在 3k 训练集上收敛稳定，最终在测试集上取得了 **83.4%** 的准确率，验证了 BERT 在中文文本分类任务上的有效性。

模型文件 `best_model.pth` 因体积较大未上传，可自行训练生成。

---
## 数据集

本项目使用今日头条新闻标题分类数据集。

- 数据来源：今日头条客户端
- 采集时间：2018 年 5 月
- 下载地址：[toutiao-text-classfication-dataset](https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset)
- 分类数量：15 类
 本项目使用的数据划分如下：

| 数据文件 | 样本数量 | 用途 |
| ------- | -------- | ---- |
| train_3k.txt | 3000 | 训练集 |
| dev_1k.txt | 1000 | 验证集 |
| test_1k.txt | 1064 | 测试集 | 

## 模型效果可视化

### 1. 模型指标
### 模型整体指标
- 总体准确率 (Accuracy): 0.8336
- 宏平均精确率 (Macro-Precision): 0.8313
- 宏平均召回率 (Macro-Recall): 0.8095
- 宏平均F1分数 (Macro-F1): 0.8179
- 最终测试集准确率: 0.8336466165413534


| 类别 | precision | recall | f1-score | support |
| :--- | :---: | :---: | :---: | :---: |
| news_agriculture | 0.7679 | 0.8113 | 0.7890 | 53 |
| news_car | 0.8774 | 0.9394 | 0.9073 | 99 |
| news_culture | 0.9143 | 0.8312 | 0.8707 | 77 |
| news_edu | 0.8608 | 0.9189 | 0.8889 | 74 |
| news_entertainment | 0.8716 | 0.8796 | 0.8756 | 108 |
| news_finance | 0.7432 | 0.7432 | 0.7432 | 74 |
| news_game | 0.7609 | 0.8750 | 0.8140 | 80 |
| news_house | 0.8864 | 0.7959 | 0.8387 | 49 |
| news_military | 0.7826 | 0.7826 | 0.7826 | 69 |
| news_sports | 0.9570 | 0.8641 | 0.9082 | 103 |
| news_story | 0.7333 | 0.6471 | 0.6875 | 17 |
| news_tech | 0.7826 | 0.7895 | 0.7860 | 114 |
| news_travel | 0.8644 | 0.8644 | 0.8644 | 59 |
| news_world | 0.7671 | 0.7568 | 0.7619 | 74 |
| stock | 0.9000 | 0.6429 | 0.7500 | 14 |
| **accuracy** | - | - | 0.8336 | 1064 |
| **macro avg** | 0.8313 | 0.8095 | 0.8179 | 1064 |
| **weighted avg** | 0.8362 | 0.8336 | 0.8336 | 1064 |

### 2. 混淆矩阵
<img width="1438" height="960" alt="811b4f5f5cef134d52702b76594b2ad1" src="https://github.com/user-attachments/assets/2229c75b-0aef-4497-949c-d2ac966b7765" />

### 3. 各类别准确率
<img width="1482" height="611" alt="9e36662888a0a6cee657a975367fd8f1" src="https://github.com/user-attachments/assets/60ee932f-6c76-4490-a3b8-a3e8dec18fa7" />

### 4. 标签分布
<img width="967" height="925" alt="3ba5c9c2db08bb8e16099c10fdcf6125" src="https://github.com/user-attachments/assets/c2010f4d-e811-4c00-aa82-2665e4dc5386" />

### 5. 模型预测置信度分布
<img width="1237" height="615" alt="697f1dcd67523ba784044fd47c17c711" src="https://github.com/user-attachments/assets/f9dc29d6-b47b-485c-adde-6a3cf70ae346" />

---

## 两组超参数对比实验
为了分析不同超参数对模型性能的影响，本项目设计了 2 组实验。所有实验均训练 12个 epoch，并使用标准交叉熵损失函数
### 实验 1
| 参数 | 取值 |
| ---- | ---- |
| 预训练模型 | bert‑base‑chinese |
| max_len | 100 |
| batch_size | 16 |
| lr | 1e‑5 |
| dropout_rate | 0.4 |
| weight_decay | 1e‑4 |
| epochs | 12 |
| num_classes | 15 |

![SwanLab1](https://github.com/user-attachments/assets/1cb09e98-05e4-4735-a334-505ba9b2f628)
<img width="1015" height="352" alt="8675df908b9bd34888380273ec9111d2" src="https://github.com/user-attachments/assets/1e74d048-ea70-40ca-901b-18650082baec" />
![SwanLab3](https://github.com/user-attachments/assets/5540f44a-57e0-4252-9d27-d00e80e8a9c4)

---

### 实验 2
| 超参数 | 数值 |
| ---- | ---- |
| 预训练模型 | bert‑base‑chinese |
| max_len | 100 |
| batch_size | 16 |
| learning_rate | 2e‑5 |
| dropout_rate | 0.3 |
| weight_decay | 1e‑4 |
| epochs | 12 |
| num_classes | 15 |
<img width="996" height="315" alt="096097c9982a784b8879626d8db91e14" src="https://github.com/user-attachments/assets/e19cee15-e350-4fa0-876a-1283f5dad279" />
<img width="1082" height="337" alt="image" src="https://github.com/user-attachments/assets/db78eae5-c368-46ca-bc27-c1d00ed9aebb" />
<img width="596" height="342" alt="image" src="https://github.com/user-attachments/assets/24e2bb5b-b852-4499-a4d5-f6c2458c2957" />


---

## 实验结论
- **实验 1（lr=1e-5 + dropout=0.4）**
  训练过程极其稳定，几乎不过拟合，验证曲线平滑，**泛化能力最优**。
- **实验2 （lr=2e-5 + dropout=0.3）**
  训练集充分收敛，但验证集 loss 震荡回升、精度剧烈波动，无法稳定收敛，**过拟合风险极高**。
-  
综上所述本实验选择实验 1（lr=1e-5 + dropout=0.4）
---

## 模型表现分析
模型整体准确率 **83.4%**，表现优秀，但存在明显的短板类别，各类别表现差异较为突出。

### 表现最差的三个类别
- news_agriculture（农业新闻）
- news_finance（财经新闻）
- stock（股票）

**原因：**
1. 样本数量极少，类别不平衡严重
2. 文本特征相似，易互相误判
3. 训练不足导致泛化能力弱

### 表现最好的三个类别
- news_car（汽车新闻）
- news_sports（体育新闻）
- news_edu（教育新闻）

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
