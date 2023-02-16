'''
Author: Echoes
Date: 2023-01-09 15:24:56
LastEditTime: 2023-02-15 20:43:06
FilePath: \lab3\data_transform.py
'''
import json
import pandas as pd

# ['Favor', 'Against', 'None'] = [0, 3, 6]
# ['PER', 'ORG', 'POL'] = [0, 1, 2]
type_dict = {
    '0F': 'PER-FAV', '0A': 'PER-AGA', '0N': 'PER-NON',
    '1F': 'ORG-FAV', '1A': 'ORG-AGA', '1N': 'ORG-NON',
    '2F': 'POL-FAV', '2A': 'POL-AGA', '2N': 'POL-NON'
}
type_list = ['PER', 'ORG', 'POL']

type = ['PER', 'ORG', 'POL']


def gen_train_data(file_path, save_path):
    with open(file_path, 'rb') as json_file:
        oneresult = json.load(json_file)
    open(save_path, 'w', encoding='utf-8')
    for i in range(len(oneresult)):

        text = oneresult[i]['text']
        if pd.isna(text):
            text = ''
        text_list = list(text)
        label_list = []

        labels = oneresult[i]['label']
        label_list = ['O' for i in range(len(text_list))]
        if len(labels) == 0:
            pass
        else:
            for label_item in labels:
                start = label_item['start']
                end = label_item['end']
                label = int(label_item['type'])
                type_ = type[label]
                label_list[start] = f'B-{type_}'
                label_list[start + 1:end - 1] = [f'I-{type_}' for i in range(end - start - 2)]
                label_list[end - 1] = f'E-{type_}'
        print(i,len(label_list),len(text_list))
        assert len(label_list) == len(text_list)
        with open(save_path, 'a',encoding='utf-8') as f:
            for idx_, line in enumerate(text_list):
                if text_list[idx_] == '\t' or text_list[idx_] == ' ':
                    text_list[idx_] = '，'
                line = text_list[idx_] + ' ' + label_list[idx_] + '\n'
                f.write(line)
            f.write("end\n")


def json2txt(inpath, outpath):
    with open(inpath, 'rb') as json_file:
        oneresult = json.load(json_file)
    open(outpath, 'w', encoding='utf-8')
    for i in range(len(oneresult)):

        text = oneresult[i]['text']
        if pd.isna(text):
            text = ''
        text_list = list(text)
        label_list = []

        labels = oneresult[i]['label']
        label_list = ['O' for i in range(len(text_list))]

        if len(labels) == 0:
            pass
        else:
            for label_item in labels:
                start = label_item['start']
                end = label_item['end']
                label = str(label_item['type']) + label_item['stance']
                type_ = type_dict[label]
                label_list[start] = f'B-{type_}'
                label_list[start + 1:end - 1] = [f'I-{type_}' for i in range(end - start - 2)]
                label_list[end - 1] = f'E-{type_}'
        print(i, len(label_list), len(text_list))
        assert len(label_list) == len(text_list)
        with open(outpath, 'a', encoding='utf-8') as f:
            for idx_, line in enumerate(text_list):
                if text_list[idx_] == '\t' or text_list[idx_] == ' ':
                    text_list[idx_] = '，'
                line = text_list[idx_] + ' ' + label_list[idx_] + '\n'
                f.write(line)
            f.write('end\n')

if __name__ == '__main__':
    gen_train_data('./data/train.json', './data/train.txt')
    gen_train_data('./data/dev.json', './data/dev.txt')
    gen_train_data('./data/test.json', './data/test.txt')
    # 拓展标签后的.txt文件，用于gtest
    json2txt('./data/test.json', './data/new_test.txt')