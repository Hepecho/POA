from data_process import *
from mtsd_model import *
import os
import time
import argparse
import pandas as pd
import sys
from transformers import BertConfig, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import classification_report
from utils import *
from data_manager import DataManager
from ner import ChineseNER
# from data_transform import json2txt

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default='BERT-LSTM', help='choose a model: BERT, BERT-LSTM')
parser.add_argument('--mode', type=str, default='train', help='choose a mode: train, utest, gtest')
args = parser.parse_args()

BERT_PATH = './bert-base-chinese'
BATCH_SIZE = 4
EPOCHS = 10
DROP_PROB = 0.5
TAGS = [
    "PER-FAV", "PER-AGA", "PER-NON",
    "ORG-FAV", "ORG-AGA", "ORG-NON",
    "POL-FAV", "POL-AGA", "POL-NON"
]
TAG_MAP = {
    'O': 0, 'START': 1, 'STOP': 2,
    "B-PER-FAV": 3, "B-PER-AGA": 4, "B-PER-NON": 5,
    "B-ORG-FAV": 6, "B-ORG-AGA": 7, "B-ORG-NON": 8,
    "B-POL-FAV": 9, "B-POL-AGA": 10, "B-POL-NON": 11,
    "I-PER-FAV": 12, "I-PER-AGA": 13, "I-PER-NON": 14,
    "I-ORG-FAV": 15, "I-ORG-AGA": 16, "I-ORG-NON": 17,
    "I-POL-FAV": 18, "I-POL-AGA": 19, "I-POL-NON": 20,
    "E-PER-FAV": 21, "E-PER-AGA": 22, "E-PER-NON": 23,
    "E-ORG-FAV": 24, "E-ORG-AGA": 25, "E-ORG-NON": 26,
    "E-POL-FAV": 27, "E-POL-AGA": 28, "E-POL-NON": 29
}  # 全局训练与测试


class mtsd():
    # Training and verification
    def train(self, model, train_iter, dev_iter):
        """
        训练模型
        :param model: 模型
        :param train_iter: 训练集数据迭代器
        :param dev_iter: 验证集数据迭代器
        """
        # 设定模型的模式为训练模式
        model.train()
        # 定义模型的损失函数
        criterion = nn.CrossEntropyLoss()
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # 设置模型参数的权重衰减
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in param_optimizer if not any(
                        nd in n for nd in no_decay)], 'weight_decay': 0.01
            }, {
                'params': [
                    p for n, p in param_optimizer if any(
                        nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # 学习率的设置
        optimizer_params = {'lr': 1e-5, 'eps': 1e-6, 'correct_bias': False}
        # 使用AdamW 主流优化器
        optimizer = AdamW(optimizer_grouped_parameters, **optimizer_params)
        # 学习率调整器，检测准确率的状态，然后衰减学习率
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, min_lr=1e-7, patience=5, verbose=True, threshold=0.0001, eps=1e-08)
        t_total = len(train_iter)
        # 设定训练轮次
        total_epochs = EPOCHS
        bestAcc = 0
        correct = 0
        total = 0
        print('Training and verification begin!')
        for epoch in range(total_epochs):
            for step, (text_list_inputids, text_list_typeids, text_list_attmask, start_list, end_list, type_list,
                       stance_list) in enumerate(train_iter):
                # 从实例化的DataLoader中取出数据，并通过 .to(device)将数据部署到服务器上
                text_list_inputids, text_list_typeids, text_list_attmask, start_list, end_list, type_list, stance_list = \
                    text_list_inputids.to(device), text_list_typeids.to(device), text_list_attmask.to(device), \
                    start_list.to(device), end_list.to(device), type_list.to(device), stance_list.to(device)
                # 梯度清零
                optimizer.zero_grad()
                # 将数据输入到模型中获得输出
                out_put = model(text_list_inputids, text_list_typeids, text_list_attmask, start_list, end_list)
                # 计算损失
                loss = criterion(out_put, stance_list)
                _, predict = torch.max(out_put.data, 1)
                correct += (predict == stance_list).sum().item()
                total += stance_list.size(0)
                loss.backward()
                optimizer.step()
                # 每两步进行一次打印
                if (step + 1) % 2 == 0:
                    train_acc = correct / total
                    print(
                        "Train Epoch[{}/{}],step[{}/{}],tra_acc{:.6f} %,loss:{:.6f}".format(
                            epoch + 1, total_epochs, step + 1, len(train_iter), train_acc * 100, loss.item()))
                # 每epoch进行一次验证
                if step + 1 == len(train_iter):
                    train_acc = correct / total
                    # 调用验证函数dev对模型进行验证，并将有效果提升的模型进行保存
                    acc = self.dev(model, dev_iter)
                    if bestAcc < acc:
                        bestAcc = acc
                        # 模型保存路径
                        torch.save(model, "./models/"+args.model+".pkl")
                    print(
                        "DEV Epoch[{}/{}],tra_acc{:.6f} %,bestAcc{:.6f}%,dev_acc{:.6f} %,loss:{:.6f}".format(
                            epoch + 1, total_epochs, train_acc * 100, bestAcc * 100, acc * 100, loss.item()))
                    model.train()  # 返回训练模式
            scheduler.step(bestAcc)

    # 验证(被train()调用)
    def dev(self, model, dev_loader, test=False):
        """
        验证集测试，被train调用或在单元评估时被调用
        :param model: 模型
        :param dev_loader: 验证集/测试集数据迭代器
        :param test: 是否输出F1等指标的控制参数
        :return: 返回正确率或指标报告（test==True）
        """
        # 设定模式为验证模式
        model.eval()
        # 设定不会有梯度的改变仅作验证
        with torch.no_grad():
            correct = 0
            total = 0
            stance_list = []
            predict_list = []
            for step, (text_list_inputids, text_list_typeids, text_list_attmask, starts, ends, types,
                       stances) in tqdm(enumerate(dev_loader), desc='Dev Itreation:'):
                text_list_inputids, text_list_typeids, text_list_attmask, starts_list, ends, types, stances = \
                    text_list_inputids.to(device), text_list_typeids.to(device), text_list_attmask.to(device), \
                    starts.to(device), ends.to(device), types.to(device), stances.to(device)
                out_put = model(text_list_inputids, text_list_typeids, text_list_attmask, starts, ends)
                _, predicts = torch.max(out_put.data, 1)
                correct += (predicts == stances).sum().item()
                stance_list.extend(stances.tolist())
                predict_list.extend(predicts.tolist())
                # print(predict)
                total += stances.size(0)

            res = correct / total
            if test:
                class_list = ['Favor', 'Against', 'None']
                report = classification_report(stance_list, predict_list, target_names=class_list, digits=4)
                return report
            return res

    # 评估
    def evaluate(self, model, dev_loader, flag='unit'):
        """
        评估模块，有unit, global两种模式选择
        :param model: 模型
        :param dev_loader: 用于评估的数据
        :param flag: 模式选择标志，default=‘unit’
        :return report: 指标报告
        """
        # 设定模式为验证模式
        model.eval()
        # 设定不会有梯度的改变仅作验证
        with torch.no_grad():
            # print(predict)
            if flag == 'unit':
                return self.dev(model, dev_loader, test=True)
            elif flag == 'global':
                stance_list = []
                predict_list = []
                for step, (text_list_inputids, text_list_typeids, text_list_attmask, starts, ends, types,
                           stances) in tqdm(enumerate(dev_loader), desc='Eva Itreation:'):
                    text_list_inputids, text_list_typeids, text_list_attmask, starts, ends, types, stances = \
                        text_list_inputids.to(device), text_list_typeids.to(device), text_list_attmask.to(device), \
                        starts.to(device), ends.to(device), types.to(device), stances.to(device)
                    out_put = model(text_list_inputids, text_list_typeids, text_list_attmask, starts, ends)
                    _, predicts = torch.max(out_put.data, 1)
                    stance_list.extend(stances.tolist())
                    predict_list.extend(predicts.tolist())
                # predict实体类型
                cn = ChineseNER("predict")
                cn.predict(inpath='./data/test.json', outpath='./data/mid_test.json')
                # 添加预测得到的立场信息predict_list，保存到文件，格式为txt
                self.savefile(predict_list, inpath='./data/mid_test.json', outpath='./data/pre_test.txt')
                # 分别将txt格式的真实和预测结果加载进来
                turedata_manager = DataManager(batch_size=BATCH_SIZE, data_type='global', tags=TAGS, tag_map=TAG_MAP,
                                               gpath='./data/new_test.txt')
                turedata_batch = turedata_manager.iteration()
                predata_manager = DataManager(batch_size=BATCH_SIZE, data_type='global', tags=TAGS, tag_map=TAG_MAP,
                                              gpath='./data/pre_test.txt')
                predata_batch = predata_manager.iteration()
                # 得到整型数组组成的元组
                _, ture_labels, _ = zip(*turedata_batch.__next__())
                _, pre_labels, _ = zip(*predata_batch.__next__())
                # print("ture:{}".format(ture_labels))
                # print("pre:{}".format(pre_labels))
                for tag in TAGS:
                    # 调用并返回utils中的定义3个评价：召回率、准确率、F1
                    f1_score(ture_labels, pre_labels, tag, TAG_MAP)

    def savefile(self, predict_list, inpath, outpath):
        """
        被evaluate调用, 生成添加了stance信息的txt文件
        :param predict_list: 立场预测结果
        :param inpath: 输入文件路径
        :param outpath: 输出文件路径
        """
        tran_dict = {
            0: 'F', 1: 'A', 2: 'N'
        }
        type_dict = {
            '0F': 'PER-FAV', '0A': 'PER-AGA', '0N': 'PER-NON',
            '1F': 'ORG-FAV', '1A': 'ORG-AGA', '1N': 'ORG-NON',
            '2F': 'POL-FAV', '2A': 'POL-AGA', '2N': 'POL-NON'
        }
        with open(inpath, 'r', encoding='utf-8') as f:  # 使用只读模型，并定义名称为f
            params = json.load(f)  # 加载json文件中的内容给params
            pre_p = 0  # predict_list指针
            for i, line in enumerate(params):
                for j, label in enumerate(line['label']):
                    params[i]['label'][j]['stance'] = tran_dict[predict_list[pre_p]]
                    pre_p += 1

        open(outpath, 'w', encoding='utf-8')
        for i in range(len(params)):

            text = params[i]['text']
            if pd.isna(text):
                text = ''
            text_list = list(text)
            labels = params[i]['label']
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
    TORCH_SEED = 21  # 随机数种子
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 设置模型在几号GPU上跑
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置device
    print(device)

    # 设置随机数种子，保证结果一致
    os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 创建数据集
    train_dataset = load_data('./data/train.json')
    dev_dataset = load_data('./data/dev.json')
    test_dataset = load_data('./data/test.json')
    train_iter = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_iter = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    Mtsd = mtsd()

    if args.mode == 'train':
        # 定义模型
        if args.model == 'BERT':
            model = BERT_Network().to(device)
        elif args.model == 'BERT-LSTM':
            model = BERT_LSTM_Network().to(device)
        else:
            raise ValueError("Unknown Model Name!")
        # 调用训练函数进行训练与验证
        start = time.time()
        Mtsd.train(model, train_iter, dev_iter)
        print("训练完毕,共用时{}秒.".format(int(time.time() - start)))
    # 单元测试
    elif args.mode == 'utest':
        loadpath = './models/BERT-LSTM.pkl'
        best_model = torch.load(loadpath, map_location=device)  # 加载最优模型
        report = Mtsd.evaluate(best_model, test_iter, flag='unit')
        print(report)
    # 全局测试
    elif args.mode == 'gtest':
        loadpath = './models/BERT-LSTM.pkl'
        best_model = torch.load(loadpath, map_location=device)  # 加载最优模型
        Mtsd.evaluate(best_model, test_iter, flag='global')
    else:
        raise ValueError("Unknown Mode!")
