from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import linecache
import argparse
import numpy as np
import torch
import pickle as pkl
import json
import os
from transformers import BertModel, BertTokenizer
from data_transform import json2txt, gen_train_data

VOCAB_PATH = './bert-base-chinese/vocab.txt'

def encoder(max_len, vocab_path, text_list):
    # 将text_list embedding成bert模型可用的输入形式
    # 加载分词模型
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    tokenizer = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors='pt'  # 返回的类型为pytorch tensor
    )
    input_ids = tokenizer['input_ids']
    token_type_ids = tokenizer['token_type_ids']
    attention_mask = tokenizer['attention_mask']
    return input_ids, token_type_ids, attention_mask


# 单元训练数据加载
def load_data(path):
    text_list = []
    target_list = []
    start_list = []
    end_list = []
    type_list = []
    stance_list = []

    # 读取数据并且处理数据
    with open(path, 'r', encoding='UTF-8') as f:
        file = json.load(f)
        for i, line in enumerate(file):
            for j, label in enumerate(line['label']):
                text_list.append(line['text'])
                # print(label[j]['labels'])
                target = label['text']
                type = label['type']
                type_list.append(type)
                start_list.append(label['start'])
                end_list.append(label['end'])
                target_list.append(target)
                stance = label['stance']
                if stance == "F":
                    stance_list.append(0)
                elif stance == "A":
                    stance_list.append(1)
                elif stance == "N":
                    stance_list.append(2)
                else:
                    print('ERROR STANCE!')
    # 调用encoder函数，获得预训练模型的三种输入形式
    text_list_inputids, text_list_typeids, text_list_attmask = encoder(
        max_len=400, vocab_path=VOCAB_PATH, text_list=text_list)
    # target_list_inputids, target_list_typeids, target_list_attmask = encoder(
        # max_len=50, vocab_path=VOCAB_PATH, text_list=target_list)
    stance_list = torch.tensor(stance_list)
    type_list = torch.tensor(type_list)
    start_list = torch.tensor(start_list)
    end_list = torch.tensor(end_list)
    # 将encoder的返回值以及label封装为Tensor的形式
    data = TensorDataset(text_list_inputids, text_list_typeids, text_list_attmask, start_list, end_list, type_list,
                         stance_list)
    return data

# 全局测试数据/单条文本加载
def load_middata(path):
    text_list = []
    target_list = []
    start_list = []
    end_list = []

    # 读取数据并且处理数据
    with open(path, 'r', encoding='UTF-8') as f:
        file = json.load(f)
        for i, line in enumerate(file):
            for j, label in enumerate(line['label']):
                text_list.append(line['text'])
                # print(label[j]['labels'])
                # type = label['labels'][0]
                target = label['text']
                # type_list.append(type)
                start_list.append(label['start'])
                end_list.append(label['end'])
                target_list.append(target)

    # 调用encoder函数，获得预训练模型的三种输入形式
    text_list_inputids, text_list_typeids, text_list_attmask = encoder(
        max_len=400, vocab_path=VOCAB_PATH, text_list=text_list)
    # target_list_inputids, target_list_typeids, target_list_attmask = encoder(
        # max_len=50, vocab_path=VOCAB_PATH, text_list=target_list)
    start_list = torch.tensor(start_list)
    end_list = torch.tensor(end_list)
    # 将encoder的返回值以及label封装为Tensor的形式

    data = TensorDataset(text_list_inputids, text_list_typeids, text_list_attmask, start_list, end_list)
    # target_data = TensorDataset(target_list_inputids, target_list_typeids, target_list_attmask, stance_list)
    return data


def _save_examples(save_dir, file_name, examples):
    count = len(examples)
    save_path = os.path.join(save_dir, file_name)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False)
    print("Save %d examples to %s." % (count, save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default="./data/data.json", type=str,
                        help="The data file exported from doccano platform.")
    parser.add_argument("--save_dir", default="./data", type=str, help="The path to save processed data.")
    parser.add_argument("--splits", default=[0.8, 0.1, 0.1], type=float, nargs="*",
                        help="The ratio of samples in datasets. [0.6, 0.2, 0.2] means 60% samples used for training, \
                        20% for evaluation and 20% for test.")
    parser.add_argument("--is_shuffle", default=True, type=bool,
                        help="Whether to shuffle the labeled dataset, defaults to True.")
    parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        raise ValueError("Please input the correct path of doccano file.")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if len(args.splits) != 3:
        raise ValueError("Only len(splits)==3 accepted for splits.")

    if args.splits and sum(args.splits) != 1:
        raise ValueError(
            "Please set correct splits, sum of elements in splits should be equal to 1."
        )

    with open(args.input_file, "r", encoding="utf-8") as f:
        raw_examples = json.load(f)
    print("total examples up to %d" % (len(raw_examples)))
    if args.is_shuffle:
        indexes = np.random.permutation(len(raw_examples))
        raw_examples = [raw_examples[i] for i in indexes]

    i1, i2, _ = args.splits
    p1 = int(len(raw_examples) * i1)
    p2 = int(len(raw_examples) * (i1 + i2))

    train_examples = raw_examples[:p1]
    dev_examples = raw_examples[p1:p2]
    test_examples = raw_examples[p2:]

    _save_examples(args.save_dir, "train.json", train_examples)
    _save_examples(args.save_dir, "dev.json", dev_examples)
    _save_examples(args.save_dir, "test.json", test_examples)

    # 转换成.txt格式用于NER
    gen_train_data('./data/train.json', './data/train.txt')
    gen_train_data('./data/dev.json', './data/dev.txt')
    gen_train_data('./data/test.json', './data/test.txt')
    # 拓展标签后的.txt文件，用于gtest
    json2txt('./data/test.json', './data/new_test.txt')
