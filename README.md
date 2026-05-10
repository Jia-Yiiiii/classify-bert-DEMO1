# classify-bert-DEMO1
基于 BERT 的今日头条新闻文本分类模型

---

##  项目介绍
本项目使用 `bert-base-chinese` 预训练模型，对 **15 类中文新闻文本**进行自动分类，实现了从数据加载、模型训练、模型评估到单句预测的完整流程，并通过 SwanLab 可视化实验过程。

本项目在 3k 训练集上，最终在测试集上取得了 **83.65%** 的准确率，验证了 BERT 在中文文本分类任务上的有效性。

模型文件 `best_model.pth` 因体积较大未上传，可自行训练生成。

---


##  模型效果可视化

### 1. 训练过程曲线
<img width="687" height="715" alt="05354f6b3a179f1955c004cd06aab7c0" src="https://github.com/user-attachments/assets/cd2ef39c-d59f-4f55-affd-c9d8eea283e0" />

### 2. 混淆矩阵
<img width="1438" height="960" alt="811b4f5f5cef134d52702b76594b2ad1" src="https://github.com/user-attachments/assets/2229c75b-0aef-4497-949c-d2ac966b7765" />


### 3. 各类别准确率
<img width="1482" height="611" alt="9e36662888a0a6cee657a975367fd8f1" src="https://github.com/user-attachments/assets/60ee932f-6c76-4490-a3b8-a3e8dec18fa7" />


### 4. 标签分布
<img width="967" height="925" alt="3ba5c9c2db08bb8e16099c10fdcf6125" src="https://github.com/user-attachments/assets/c2010f4d-e811-4c00-aa82-2665e4dc5386" />


### 5. 模型预测置信度分布
<img width="1237" height="615" alt="697f1dcd67523ba784044fd47c17c711" src="https://github.com/user-attachments/assets/f9dc29d6-b47b-485c-adde-6a3cf70ae346" />

### 6. SwanLab 可视化
## 超参数配置
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
<img width="985" height="345" alt="4a3b3b667c5ee35d1aa94cc78a08d763" src="https://github.com/user-attachments/assets/1cb09e98-05e4-4735-a334-505ba9b2f628" />
<img width="1015" height="352" alt="8675df908b9bd34888380273ec9111d2" src="https://github.com/user-attachments/assets/d6a72bde-6eb1-461e-99c7-9e471e22ceb5" />
<img width="482" height="342" alt="551cf303aee6e96cfc4e6865d8eb321f" src="https://github.com/user-attachments/assets/5540f44a-57e0-4252-9d27-d00e80e8a9c4" />

## 超参数配置
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


---

## 项目结构
```text
classify-bert-DEMO1/
├── DATA/                # 数据集文件夹
│   ├── .gitkeep
│   ├── train_3k.txt     # 训练集
│   ├── dev_1k.txt       # 验证集
│   └── test_1k.txt      # 测试集
├── config/              # 配置文件目录
│   ├── .gitkeep
│   ├── Bert_Config_exp1.json      # 模型超参数配置
│   └── label_map.json   # 标签映射文件
├── model.py             # 模型定义文件
├── Predict.py           # 单句/批量文本分类预测程序
├── README.md            # 项目说明文档
├── requirements.txt     # 依赖包列表
├── trainer.py           # 训练主程序
└── utils.py             # 工具函数

```
### 快速开始
pip install -r requirements.txt
