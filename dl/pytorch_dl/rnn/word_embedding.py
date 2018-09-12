import torch
from torch import nn
from torch.autograd import Variable

# 定义词嵌入
embeds = nn.Embedding(2, 5)  # 2 个单词，维度 5
print(embeds.weight)

embeds.weight.data = torch.ones(2, 5)
embeds.weight
print(embeds.weight)

embeds = nn.Embedding(10, 10)
single_word_embed = embeds(Variable(torch.LongTensor([5])))

print(embeds.weight)
print(single_word_embed)


ll = [(('When', 'forty'), 'winters'), (('forty', 'winters'), 'shall')]

word , label = ll[1]
print(word, label)