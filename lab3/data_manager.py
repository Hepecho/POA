import copy
import pickle as cPickle    # pickle库，读取序列化
import torch

class DataManager():
    # 初始化，进行训练train
    def __init__(self, gpath='', tag_map={}, max_length=400, batch_size=20, data_type='train', tags=[]):
        self.index = 0
        self.input_size = 0
        self.batch_size = batch_size
        self.max_length = max_length
        self.data_type = data_type
        self.data = []
        self.batch_data = []
        self.vocab = {"unk": 0}
        # self.tag_map = {"O":0, "B-ORG":1, "I-ORG":2, "E-ORG":3, "B-PER":4, "I-PER":5, "E-PER":6, "S":7}
        self.tag_map = {"O": 0, "START": 1, "STOP": 2}

        # 训练集
        if data_type == "train":
            assert tags, Exception("请指定需要训练的tag类型，如[\"ORG\", \"PER\"]")     # 若没有tag，触发断言提示
            self.generate_tags(tags)
            self.data_path = "data/train.txt"
        # 验证集
        elif data_type == "dev":
            self.data_path = "data/dev.txt"
            self.load_data_map()
        # 测试集
        elif data_type == "test":
            self.data_path = "data/test.txt"
            self.load_data_map()
        elif data_type == "global":
            self.generate_tags(tags)
            self.tag_map = tag_map
            self.data_path = gpath
            with open("models/data.pkl", "rb") as f:
                # 使用cPickle加载序列化数据
                self.data_map = cPickle.load(f)
                # 从data_map中获取vocab
                self.vocab = self.data_map.get("vocab", {})

        # 加载数据，分batch
        self.load_data()
        self.prepare_batch()

    # 生成标签tags
    def generate_tags(self, tags):
        self.tags = []
        for tag in tags:
            for prefix in ["B-", "I-", "E-"]:
                self.tags.append(prefix + tag)
        self.tags.append("O")

    # 加载数据图data.pkl
    def load_data_map(self):
        with open("models/data.pkl", "rb") as f:
            # 使用cPickle加载序列化数据
            self.data_map = cPickle.load(f)
            # 从data_map中获取三个属性
            self.vocab = self.data_map.get("vocab", {})
            self.tag_map = self.data_map.get("tag_map", {})
            self.tags = self.data_map.keys()

    # 加载数据
    def load_data(self):
        # load data
        # add vocab
        # covert to one-hot
        sentence = []   # 列表存放word
        target = []     # 列表存放tag
        with open(self.data_path, encoding='utf-8') as f:
            for line in f:
                line = line[:-1]
                # 遇到end，本句结束
                if line == "end":
                    self.data.append([sentence, target])
                    sentence = []
                    target = []
                    continue
                try:
                    word, tag = line.split(" ")
                except Exception:
                    continue

                # 词典里没有这个词，加入词典
                if word not in self.vocab and self.data_type == "train":
                    self.vocab[word] = max(self.vocab.values()) + 1
                # 标签集里没有这个tag，加入标签集
                if tag not in self.tag_map and self.data_type == "train" and tag in self.tags:
                    self.tag_map[tag] = len(self.tag_map.keys())

                sentence.append(self.vocab.get(word, 0)) 
                target.append(self.tag_map.get(tag, 0))

        self.input_size = len(self.vocab.values())      # input_size = 词典长度

        print("{} data: {}".format(self.data_type, len(self.data)))     # 数据总长度
        print("vocab size: {}".format(self.input_size))                 # 词典长度
        print("unique tag: {}".format(len(self.tag_map.values())))      # 标签集长度
        print("-"*50)

    # 转换标签
    def convert_tag(self, data):
        # add E-XXX for tags
        # add O-XXX for tags
        _, tags = data
        converted_tags = []
        for _, tag in (tags[:-1]):
            if tag not in self.tag_map and self.data_type == "train":
                self.tag_map[tag] = len(self.tag_map.keys())
            converted_tags.append(self.tag_map.get(tag, 0))
        converted_tags.append(0)
        data[1] = converted_tags
        assert len(converted_tags) == len(tags), "convert error, the list dosen't match!"
        return data

    # 准备 batch
    def prepare_batch(self):
        '''
            prepare data for batch
        '''
        index = 0
        while True:
            if index+self.batch_size >= len(self.data):
                pad_data = self.pad_data(self.data[-self.batch_size:])
                self.batch_data.append(pad_data)
                break
            else:
                pad_data = self.pad_data(self.data[index:index+self.batch_size])
                index += self.batch_size
                self.batch_data.append(pad_data)
    
    def pad_data(self, data):
        c_data = copy.deepcopy(data)
        # print(c_data)
        max_length = max([len(i[0]) for i in c_data])
        for i in c_data:
            i.append(len(i[0]))
            i[0] = i[0] + (max_length-len(i[0])) * [0]
            i[1] = i[1] + (max_length-len(i[1])) * [0]
            # i[0] = torch.tensor(i[0])
            # i[1] = torch.tensor(i[1])
        return c_data

    # 迭代
    def iteration(self):
        idx = 0
        while True:
            yield self.batch_data[idx]
            idx += 1
            if idx > len(self.batch_data)-1:
                idx = 0

    def get_batch(self):
        for data in self.batch_data:
            yield data
