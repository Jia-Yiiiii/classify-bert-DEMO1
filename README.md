# classify-bert-DEMO1
基于 BERT 的今日头条新闻文本分类模型

---

## 项目结构
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
