import streamlit as st
import json
import os
import torch
import torch.optim as optim     # torch.optim：优化算法库
import time
import pickle   # pickle库，序列化
import sys
import yaml     # yaml库，关于yml格式文件的操作
import pandas as pd
import numpy as np
from tqdm import tqdm

from ner import ChineseNER
from ner_model import BiLSTMCRF
from data_manager import DataManager
from data_process import load_middata
from torch.utils.data import Dataset, DataLoader, TensorDataset

st.set_page_config(page_title="Demo", initial_sidebar_state="auto", layout="wide")

Ans = {
    "text": "",
    "target": [],
    "type": [],
    "stance": [],
    "position": []
}

# 初始化Ans，将NER部分得到的信息装入Ans
def init_Ans(path):
    with open(path, 'r', encoding='utf-8') as f:  # 使用只读模型，并定义名称为f
        params = json.load(f)  # 加载json文件中的内容给params
        Ans['text'] = params[0]['text']
        for j, label in enumerate(params[0]['label']):
            # type = int(params[i]['label'][j]['labels'][0][0])
            Ans['target'].append(label['text'])
            Ans['type'].append(label['type'])
            Ans['position'].append((label['start'], label['end']))


def predict(input, model, device):
    # 设定模式为验证模式
    model.eval()
    pre_list = []
    # 设定不会有梯度的改变仅作验证
    with torch.no_grad():
        for step, (text_list_inputids, text_list_typeids, text_list_attmask, start_list, end_list) in tqdm(
                enumerate(input), desc='Dev Itreation:'):
            text_list_inputids, text_list_typeids, text_list_attmask, start_list, end_list = text_list_inputids.to(
                device), text_list_typeids.to(device), text_list_attmask.to(device), start_list.to(device), end_list.to(
                device)
            out_put = model(text_list_inputids, text_list_typeids, text_list_attmask, start_list, end_list)
            print(start_list)
            _, predict = torch.max(out_put.data, 1)
            predict = predict.numpy().tolist()
            # print(predict)
            pre_list.extend(predict)
        if len(Ans['target']) != len(pre_list):
            print("不一致的列表长度！")
            print(Ans['target'])
            print(pre_list)
        else:
            Ans['stance'] = pre_list


def writer():
    # st.markdown('Demo')
    st.title('Demo')
    text = st.text_input('请输入待分析的文本内容')
    text = "[{\"text\":\"" + text +"\"}]"
    print(text)
    jsontext = json.loads(text)
    raw = open("./data/input.json", 'w')
    json.dump(jsontext, raw)
    raw.close()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 得到输入input.json
    if st.button("分析"):
        st.write('正在分析……')
        # 加载NER模型
        ner_model = BiLSTMCRF().to(device)
        # predict实体类型
        cn = ChineseNER("predict")
        cn.predict()
        # 根据cn输出的临时文件初始化Ans
        midtext_path = './data/result.json'
        init_Ans(midtext_path)

        # 加载MTSD模型
        loadpath = './models/BERT-LSTM.pkl'
        mtsd_model = torch.load(loadpath, map_location=torch.device('cpu'))
        # 加载cn输出的临时文件result.json
        test_dataset = load_middata(midtext_path)
        test_iter = DataLoader(test_dataset, batch_size=20, shuffle=False)
        # print(text)
        # predict立场
        predict(test_iter, mtsd_model, device)
        # 终端输出
        print(len(Ans['target']))
        print('Ans is : {}'.format(Ans))
        for i in range(len(Ans['target'])):
            if Ans['type'][i] == 0:
                Ans['type'][i] = '人物'
            elif Ans['type'][i] == 1:
                Ans['type'][i] = '组织'
            elif Ans['type'][i] == 2:
                Ans['type'][i] = '政策'
            if Ans['stance'][i] == 0:
                Ans['stance'][i] = 'Favor'
            elif Ans['stance'][i] == 1:
                Ans['stance'][i] = 'Against'
            elif Ans['stance'][i] == 2:
                Ans['stance'][i] = 'None'
        print('Ans is : {}'.format(Ans))
        # 网页显示结果
        tb = pd.DataFrame(Ans)
        st.table(tb)
        st.stop()
    else:
        st.stop()
    st.write('The current movie title is', text)


if __name__ == '__main__':
    writer()
