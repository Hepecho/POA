import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import transformers


class BERT_Network(nn.Module):
	def __init__(self):
		super(BERT_Network, self).__init__()
		# 加载预训练模型
		pretrained_weights = "./bert-base-chinese/"
		self.bert = transformers.BertModel.from_pretrained(pretrained_weights)
		for param in self.bert.parameters():
			param.requires_grad = True
		self.lstm = nn.GRU(input_size=768, hidden_size=384, batch_first=True, bidirectional=True)
		self.dense = nn.Linear(768, 3)  # bert默认的隐藏单元数是768， 输出单元3，表示三分类

	def forward(self, text_list_inputids, text_list_typeids, text_list_attmask, start_list, end_list):
		# 得到bert_output
		text_output = self.bert(
			input_ids=text_list_inputids, token_type_ids=text_list_typeids, attention_mask=text_list_attmask)

		token_msg = []
		cls_msg = text_output[1].unsqueeze(1)  # [batch, 1, 768]

		for i, pos in enumerate(zip(start_list, end_list)):
			token = text_output[0][i, pos[0]+1:pos[1]+1:]
			# print("token:{}".format(token.shape))
			token_msg.append(token)
		token_msg_ = pad_sequence(token_msg, batch_first=True, padding_value=0)
		input_cat = torch.cat((cls_msg, token_msg_), dim=-2)  # 将句子和aspect按x维度拼接

		linear_output = self.dense(input_cat[:, -1, :])
		# [batch, 3]
		return linear_output

class BERT_LSTM_Network(nn.Module):
	def __init__(self):
		super(BERT_LSTM_Network, self).__init__()
		# 加载预训练模型
		pretrained_weights = "./bert-base-chinese/"
		self.bert = transformers.BertModel.from_pretrained(pretrained_weights)
		for param in self.bert.parameters():
			param.requires_grad = True
		self.lstm = nn.GRU(input_size=768, hidden_size=384, batch_first=True, bidirectional=True)
		self.dense = nn.Linear(768, 3)  # bert默认的隐藏单元数是768， 输出单元3，表示三分类

	def forward(self, text_list_inputids, text_list_typeids, text_list_attmask, start_list, end_list):
		text_output = self.bert(
			input_ids=text_list_inputids, token_type_ids=text_list_typeids, attention_mask=text_list_attmask)
		token_msg = []
		cls_msg = text_output[1].unsqueeze(1)  # [batch, 1, 768]

		for i, pos in enumerate(zip(start_list, end_list)):
			token = text_output[0][i, pos[0]+1:pos[1]+1:]
			token_msg.append(token)
		token_msg_ = pad_sequence(token_msg, batch_first=True, padding_value=0)
		input_cat = torch.cat((cls_msg, token_msg_), dim=-2)  # 将句子和aspect按x维度拼接 [batch, maxlen+1, 768]
		out_put, _ = self.lstm(input_cat)
		linear_output = self.dense(out_put[:, -1, :])  # out_put[:, -1, :]: [batch, 768]
		return linear_output  # linear_output [batch, 3]
