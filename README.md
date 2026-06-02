# classify-bert-DEMO1
基于 BERT 的今日头条新闻文本分类模型

---

## 项目介绍
本项目使用 `bert-base-chinese` 预训练模型，对 **15 类中文新闻文本**进行自动分类，实现了从数据加载、模型训练、模型评估到单句预测的完整流程，并通过 SwanLab 可视化实验过程。

模型在 3k 训练集上收敛稳定，最终在测试集上取得了 **84.39%** 的准确率，验证了 BERT 在中文文本分类任务上的有效性。

模型文件 `best_model.pth` 因体积较大未上传，可自行训练生成。

---

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
---

---
### 1. 模型指标
### 模型整体指标
<img width="648" height="589" alt="7aaad8c75676b1184ea45384a4d55bd8" src="https://github.com/user-attachments/assets/9a267544-402f-407a-b2e6-a08bff7c0555" />
## 单句文本预测演示
支持输入任意中文新闻文本，直接输出分类结果，示例：
- 输入文本：`神舟十八号载人飞船成功发射，圆满完成任务！`
- 预测类别：`news_military`（军事类）

- <img width="278" height="72" alt="f432ebc652243bb3eb6b9b01c2234f47" src="https://github.com/user-attachments/assets/2bec1acf-22da-4249-b539-6753a2f6ee9a" />


<img width="462" height="65" alt="image" src="https://github.com/user-attachments/assets/74abaeed-35f6-4a9e-8d0a-3c1da4994049" />

### 使用方式
运行 `Predict.py`，修改代码内 `demo_text` 变量即可自定义输入文本进行推理。
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
---
---
## 环境配置
使用 Conda 创建独立环境，版本严格匹配：
```bash
conda create -n bert-text-cls python=3.6
conda activate bert-text-cls
torch==1.10.2
transformers==4.16.2
scikit-learn==0.19.1
tqdm==4.64.1
numpy==1.19.5
