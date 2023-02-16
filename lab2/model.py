import torch
import torch.nn as nn

# LSTM模型
class LSTM_Network(nn.Module):
	def __init__(self, emb_matrix):
		super(LSTM_Network, self).__init__()
		self.embedding = nn.Embedding.from_pretrained(emb_matrix, freeze=False)  # embedding层
		# LSTM层，同学们自行调参，如果bidirectional设置True则为BiLSTM，但是相应的hidden_size记得除2
		self.lstm = nn.GRU(input_size=300, hidden_size=150, batch_first=True, bidirectional=True)
		self.fc = nn.Linear(in_features=300, out_features=3)  # 全连接层

	def forward(self, input):
		input = self.embedding(input)
		# print(input.shape)
		output, _ = self.lstm(input)  # output的size为[batch, seq, embedding]
		# print(output.shape)
		output = self.fc(output[:, -1, :])  # 句子最后时刻的 hidden state

		return output

# AE-LSTM模型
class AELSTM_Network(nn.Module):
	def __init__(self, emb_matrix):
		super(AELSTM_Network, self).__init__()
		self.embedding = nn.Embedding.from_pretrained(emb_matrix, freeze=False)
		self.lstm = nn.GRU(input_size=300 * 2, hidden_size=150, batch_first=True, bidirectional=True)
		self.fc = nn.Linear(in_features=300, out_features=3)

	def forward(self, input, aspect):
		input = self.embedding(input)
		batch_size, seq_len, emb_ = input.size()
		aspect = self.embedding(aspect)
		aspect = aspect.unsqueeze(1).expand(-1, seq_len, -1)
		input = torch.cat((input, aspect), dim=-1)  # 将句子和aspect按最后一维度拼接
		output, _ = self.lstm(input)  # output的size为[batch, seq, embedding]
		output = self.fc(output[:, -1, :])  # 句子最后时刻的 hidden state

		return output

# ATAE—LSTM模型
class ATAELSTM_Network(nn.Module):
	def __init__(self, emb_matrix):
		super(ATAELSTM_Network, self).__init__()
		self.embedding = nn.Embedding.from_pretrained(emb_matrix, freeze=False)
		self.lstm = nn.GRU(input_size=300 * 2, hidden_size=150, batch_first=True, bidirectional=True)

		self.fc = nn.Linear(in_features=300 * 2, out_features=3)

		# 用于attention的参数
		self.linear = nn.Linear(300, 300, bias=False)
		self.tanh = nn.Tanh()  # 激活函数
		self.softmax = nn.Softmax(dim=-1) # softmax

		self.w = nn.Parameter(torch.Tensor(300, 1))
		self.w.data.normal_(mean=0.0, std=0.02)

	def forward(self, input, aspect, mask):
		input = self.embedding(input)
		batch_size, seq_len, emb_ = input.size()
		aspect = self.embedding(aspect)
		aspect = aspect.unsqueeze(1).expand(-1, seq_len, -1)
		# [25, len, 300]

		input = torch.cat((input, aspect), dim=-1)  # 将句子和aspect按最后一维度拼接
		# [batch, seq, embedding] [25, len, 600]
		output, _ = self.lstm(input)
		# [batch, seq, embedding] [25, len, 300]

		output_2 = self.attention(output, mask)  # attention机制
		# [batch, emb_size] [25, 300]

		output = torch.cat((output_2, output[:, -1, :]), dim=-1)  # 论文中的操作，将attention后的表示和lstm最后的hidden进行拼接
		print(output.shape)
		output = self.tanh(self.fc(output))

		return output

	# attention机制，有很多attention的操作大家可以自行百度，如global_attention, local_attention, multi-head_attention等
	def attention(self, K, mask=None):
		batch_size, seq_len, emb_size = K.size()
		# k.shape = [batch, seq, embedding] [25, len, 300]

		K_ = K.reshape(-1, emb_size)
		# k_.shape = [batch*seq, emb] [25*len, 300]
		K_ = self.linear(K_)
		# [25*len, 300]
		K_ = self.tanh(K_)

		attention_matrix = torch.mm(K_, self.w).view(batch_size, -1)  # 矩阵乘法[25*len, 300]*[300,1] -> [25, len]

		if mask is not None:  # mask矩阵我没有生成，有兴趣的同学可以自己生成然后重新跑跑模型看结果变化情况。
			# mask = mask.bool()
			# [25， len]
			attention_matrix.masked_fill_(mask == 0, -float('inf'))

		attention_matrix = self.softmax(attention_matrix)  # 生成attention矩阵
		output = torch.bmm(attention_matrix.unsqueeze(1), K)  # torch.bmm 忽略batch维的矩阵乘法
		# [batch, 1, emb_size] [25, 1, 300]
		output = output.squeeze(1)  # [batch, emb_size] [25, 300]

		return output








