from data_process import *
from model import *
import os
import torch.nn as nn
import argparse
import pandas as pd


parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default='LSTM', required=True, help='choose a model: LSTM, TextRNN, AELSTM, ATAELSTM')
args = parser.parse_args()
cache = {
        'train_loss': []
    }
# 训练函数

def train(epochs):
    model.train()  # 模型设置成训练模式
    for epoch in range(epochs):  # 训练epochs轮
        loss_sum = 0  # 记录每轮loss
        for batch in train_iter:
            input_, aspect, label = batch
            # print(input_)
            mask_ = input_.bool()
            # print('mask_:' + str(mask_.shape))
            # print(mask_)
            optimizer.zero_grad()  # 每次迭代前设置grad为0

            # 不同的模型输入不同，请同学们看model.py文件
            # output = model(input_)
            output = model(input_, aspect, mask_)

            loss = criterion(output, label)  # 计算loss
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            loss_sum += loss.item()  # 累积loss
        loss_current = loss_sum / len(test_iter)
        cache['train_loss'].append(float(loss_current))
        print('epoch: ', epoch, 'loss:', loss_current)

    test_acc = evaluate()  # 模型训练完后进行测试
    print('test_acc:', test_acc)
    return cache


# 测试函数
def evaluate():
    model.eval()
    total_acc, total_count = 0, 0
    loss_sum = 0

    with torch.no_grad():  # 测试时不计算梯度
        for batch in test_iter:
            input_, aspect, label = batch

            mask_ = input_.bool()
            # print(mask_.shape)
            # predicted_label = model(input_)
            predicted_label = model(input_, aspect, mask_)

            loss = criterion(predicted_label, label)  # 计算loss
            total_acc += (predicted_label.argmax(1) == label).sum().item()  # 累计正确预测数
            total_count += label.size(0)  # 累积总数
            loss_sum += loss.item()  # 累积loss
        print('test_loss:', loss_sum / len(test_iter))

    return total_acc / total_count


if __name__ == '__main__':

    TORCH_SEED = 21  # 随机数种子
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 设置模型在几号GPU上跑
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置device

    # 设置随机数种子，保证结果一致
    os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 创建数据集
    train_dataset = MyDataset('./data/acsa_train.json')
    test_dataset = MyDataset('./data/acsa_test.json')
    train_iter = DataLoader(train_dataset, batch_size=25, shuffle=True, collate_fn=batch_process)
    test_iter = DataLoader(test_dataset, batch_size=25, shuffle=False, collate_fn=batch_process)

    # 加载我们的Embedding矩阵
    embedding = torch.tensor(np.load('./emb/my_embeddings.npz')['embeddings'], dtype=torch.float)

    # 定义模型
    if args.model == 'LSTM':
        model = LSTM_Network(embedding).to(device)
    elif args.model == 'AELSTM':
        model = AELSTM_Network(embedding).to(device)
    else:
        model = ATAELSTM_Network(embedding).to(device)

    # 定义loss函数、优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, weight_decay=0.001)

    # 开始训练
    cache = train(40)

    # cache['train_loss'].append(1.14514)
    path = os.path.join('results', args.model)
    # print(cache)

    colums = ['train_loss']
    data = cache['train_loss']
    print(colums, data[-1])

    save = pd.DataFrame(columns=colums, data=data)

    # changed
    # if os.path.exists(path):
    #     os.remove(path)
    f1 = open(path+'.csv', mode='w', newline='')
    save.to_csv(f1, encoding='gbk')
    f1.close()

    # ATAE LSTM
    # test_loss: 0.5981685056978342
    # test_acc: 0.7963726298433635

    # ATAE-BiLSTM
    # test_loss: 0.622771796219203
    # test_acc: 0.786479802143446

    # ATAEM-BiLSTM
    # test_loss: 0.6695821434259415
    # test_acc: 0.7642209398186315

    # BiLSTM
    # test_loss: 0.6721391895291756
    # test_acc: 0.7411376751854906

    # AE-BiLSTM
    # test_loss: 0.8350130651070147
    # test_acc: 0.6248969497114591
