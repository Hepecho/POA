# README
***
## 1. 简介
这是舆情分析综合实验实体-立场任务的项目代码，实现思路采用*pipeline*，分为两个模块：*实体识别*和*基于多个实体的立场分析*，即NER+MTSD
## 2. 文件结构
```
|-- README.md
|-- bankup  # 存放备份代码和数据  
|-- bert-base-chinese  # 预训练BERT模型
|-- data  # 数据 
|   |-- data.json  # 预处理后的数据集（格式用于MTSD）
|   |-- data.txt  # 预处理后的数据（格式用于NER）
|   |-- mid_test.json  # gtest模式或predict模式下产生的临时文件
|   |-- pre_test.json  # gtest模式下产生的临时文件
|   |-- train.json  # 以下六个文件由data.json分割并转换而来
|   |-- train.txt
|   |-- dev.json
|   |-- dev.txt
|   |-- test.json
|   |-- test.txt
|   |-- new_test.txt  # 拓展标签后的test.txt，用于gtest
|   |-- input.json  # GUI测试产生的临时文件
|   `-- result.json  # GUI测试产生的临时文件
|-- models  # 存放训练好的模型
|-- app.py  # GUI展示
|-- data_manager.py  # NER数据预处理
|-- data_process.py  # MTSD数据预处理
|-- data_transform.py # MSTD数据格式转换NER数据格式
|-- main.py  # MSTD模型训练、评估和测试
|-- mtsd_model.py  # MTSD模型定义
|-- ner.py  # NER模型训练和评估 
|-- ner_model.py  # NER模型定义
|-- utils.py  # NER部分方法和评估指标计算
```
## 3. 用法


NER模型单元训练、测试
```
python ner.py train
python ner.py predict  # 生成包含实体类别的mid_test.json文件
```
MTSD模型训练
```
python main.py --mode train
```
MTSD模型单元测试
```
python main.py --mode utest
```
项目整体性能测试
```
python main.py --mode gtest
```
运行Demo网页
```
streamlit run app.py
or
streamlit run app.py --server.port your_port
```
重新分割数据集(8:1:1分割为例)
```
python data_process.py --splits 0.8 0.1 0.1
```
