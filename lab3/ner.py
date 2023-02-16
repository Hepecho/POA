# -- coding: utf-8 --
import pickle   # pickle库，序列化
import sys
import yaml     # yaml库，关于yml格式文件的操作
import json

import torch
import torch.optim as optim     # torch.optim：优化算法库

import matplotlib.pyplot as plt

from data_manager import DataManager
from ner_model import BiLSTMCRF
from utils import f1_score, get_tags, format_result


# TAG_0 = ["ORG", "PER", "POL"]  # 单元训练与测试

class ChineseNER(object):
    '''
    提供两种运行形式：训练train 和 预测predict
    训练：python main.py train
    预测：python main.py predict
    '''

    # 初始化
    def __init__(self, entry="train"):
        self.load_config()
        self.__init_model(entry)
    # 初始化模型（train和predict）

    def __init_model(self, entry):
        # 形式1：训练train
        if entry == "train":
            self.train_manager = DataManager(batch_size=self.batch_size, tags=self.tags)
            self.total_size = len(self.train_manager.batch_data)
            data = {
                "batch_size": self.train_manager.batch_size,
                "input_size": self.train_manager.input_size,
                "vocab": self.train_manager.vocab,
                "tag_map": self.train_manager.tag_map,
            }
            self.save_params(data)
            dev_manager = DataManager(batch_size=30, data_type="dev")
            self.dev_batch = dev_manager.iteration()

            self.model = BiLSTMCRF(
                tag_map=self.train_manager.tag_map,
                batch_size=self.batch_size,
                vocab_size=len(self.train_manager.vocab),
                dropout=self.dropout,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size,
            )

            self.restore_model()

        # 形式2：预测predict
        elif entry == "predict":
            data_map = self.load_params()
            input_size = data_map.get("input_size")
            self.tag_map = data_map.get("tag_map")
            self.vocab = data_map.get("vocab")

            self.model = BiLSTMCRF(
                tag_map=self.tag_map,
                vocab_size=input_size,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size
            )

            self.restore_model()

    # 加载参数
    def load_config(self):
        try:
            fopen = open("models/config.yml")
            config = yaml.safe_load(fopen)
            fopen.close()
        except Exception as error:
            print("Load config failed, using default config {}".format(error))
            fopen = open("models/config.yml", "w")
            config = {
                "embedding_size": 100,
                "hidden_size": 256,
                "batch_size": 8,
                "dropout": 0.5,
                "model_path": "models/",
                "tags": ["ORG", "PER", "POL"]
            }
            yaml.dump(config, fopen)
            fopen.close()
        self.embedding_size = config.get("embedding_size")
        self.hidden_size = config.get("hidden_size")
        self.batch_size = config.get("batch_size")
        self.model_path = config.get("model_path")
        self.tags = config.get("tags")
        self.dropout = config.get("dropout")

    # 恢复模型
    def restore_model(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path + "params.pkl"))
            print("model restore success!")
        except Exception as error:
            print("model restore faild! {}".format(error))

    # 保存（写入）
    def save_params(self, data):
        with open("models/data.pkl", "wb") as fopen:
            pickle.dump(data, fopen)
    # 加载（读出）
    def load_params(self):
        with open("models/data.pkl", "rb") as fopen:
            data_map = pickle.load(fopen)
        return data_map

    # 模式1：训练train
    def train(self):
        # 两种可选的优化方式：Adam 和 SGD
        optimizer = optim.Adam(self.model.parameters())
        # optimizer = optim.SGD(ner_model.parameters(), lr=0.01)

        for epoch in range(20):    # epoch = 100
            index = 0
            for batch in self.train_manager.get_batch():
                index += 1
                self.model.zero_grad()

                sentences, tags, length = zip(*batch)
                sentences_tensor = torch.tensor(sentences, dtype=torch.long)
                tags_tensor = torch.tensor(tags, dtype=torch.long)
                length_tensor = torch.tensor(length, dtype=torch.long)

                loss = self.model.neg_log_likelihood(sentences_tensor, tags_tensor, length_tensor)
                progress = ("█"*int(index * 25 / self.total_size)).ljust(25)
                print("""epoch [{}] |{}| {}/{}\n\tloss {:.2f}""".format(
                        epoch, progress, index, self.total_size, loss.cpu().tolist()[0]
                    )
                )
                self.evaluate()     # 调用评价函数
                print("-"*50)
                loss.backward()
                optimizer.step()
                # pytorch把所有的模型参数用一个内部定义的dict进行保存，自称为“state_dict”——state_dict即不带模型结构的模型参数
                # torch.save将模型参数state_dict，保存入params.pkl文件
                torch.save(self.model.state_dict(), self.model_path+'params.pkl')

    # train中的评价函数evaluate
    def evaluate(self):
        sentences, labels, length = zip(*self.dev_batch.__next__())
        _, paths = self.model(sentences)
        print("\teval")
        for tag in self.tags:
            # 调用并返回utils中的定义3个评价：召回率、准确率、F1
            # print(len(labels))
            # print(len(paths))
            # print(self.model.tag_map)
            f1_score(labels, paths, tag, self.model.tag_map)

    # 模式2：预测predict
    def predict(self, inpath='./data/input.json', outpath='./data/result.json'):
        with open(inpath, 'r', encoding="utf-8") as file:
            oneresult = json.load(file)

        for i in range(len(oneresult)):
            text = oneresult[i]['text']

            input_vec = [self.vocab.get(i, 0) for i in text]
            # convert to tensor
            sentences = torch.tensor(input_vec).view(1, -1)
            _, paths = self.model(sentences)

            entities = []
            for tag in self.tags:
                tags = get_tags(paths[0], tag, self.tag_map)
                entities += format_result(tags, text, tag)

            data = {"text": text, "label": entities}

            if i == 0:
                json_str = json.dumps(data, ensure_ascii=False, indent=4)
                with open(outpath, 'w', encoding='utf-8') as json_file:
                    json_file.write("[")
                    json_file.write(json_str)
                    if i!=len(oneresult)-1:
                        json_file.write(",")
                    else:
                        json_file.write("]")
            elif i == len(oneresult)-1:  # 能到这个分支不？
                json_str = json.dumps(data, ensure_ascii=False, indent=4)
                with open(outpath, 'a', encoding='utf-8') as json_file:
                    json_file.write(json_str)
                    json_file.write("]")
            else:
                json_str = json.dumps(data, ensure_ascii=False, indent=4)
                with open(outpath, 'a', encoding='utf-8') as json_file:
                    json_file.write(json_str)
                    if i != len(oneresult) - 1:
                        json_file.write(",")
                    else:
                        json_file.write("]")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = BiLSTMCRF().to(device)

    if len(sys.argv) < 2:
        print("menu:\n\ttrain\n\tpredict")
        exit()
    if sys.argv[1] == "train":
        cn = ChineseNER("train")
        cn.train()
    elif sys.argv[1] == "predict":
        cn = ChineseNER("predict")
        cn.predict(inpath='./data/test.json', outpath='./data/mid_test.json')
