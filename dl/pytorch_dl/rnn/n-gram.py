import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

CONTEXT_SIZE = 2  # 依据的单词数
EMBEDDING_DIM = 10  # 词向量的维度
# 我们使用莎士比亚的诗
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

trigram = [((test_sentence[i], test_sentence[i + 1]), test_sentence[i + 2])
           for i in range(len(test_sentence) - 2)]

print("trigram:", len(trigram), trigram)

# 建立每个词与数字的编码，据此构建词嵌入
vocb = set(test_sentence)  # 使用 set 将重复的元素去掉
print("vocb:", len(vocb), vocb)

word_to_idx = {word: i for i, word in enumerate(vocb)}
print(word_to_idx)

idx_to_word = {word_to_idx[word]: word for word in word_to_idx}
print(idx_to_word)


# OK
# The network
class NGram(nn.Module):
    def __init__(self, vocab_size, context_size=CONTEXT_SIZE, n_dim=EMBEDDING_DIM):
        super(NGram, self).__init__()
        self.embed = nn.Embedding(vocab_size, n_dim)
        self.classify = nn.Sequential(
            nn.Linear(context_size * n_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, vocab_size)
        )

    def forward(self, x):
        vob_embed = self.embed(x)
        vob_embed = vob_embed.view(1, -1)
        out = self.classify(vob_embed)
        return out


net = NGram(len(word_to_idx))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-5)

for e in range(100):
    train_loss = 0
    for word, label in trigram:  # 使用前 100 个作为训练集
        word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))  # 将两个词作为输入
        label = Variable(torch.LongTensor([word_to_idx[label]]))
        # 前向传播
        out = net(word)
        loss = criterion(out, label)
        train_loss += loss.item()
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (e + 1) % 20 == 0:
        print('epoch: {}, Loss: {:.6f}'.format(e + 1, train_loss / len(trigram)))

net = net.eval()

# 测试一下结果
word, label = trigram[75]
print('input: {}'.format(word))
print('label: {}'.format(label))
print()
word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
out = net(word)
pred_label_idx = out.max(1)[1].item()
predict_word = idx_to_word[pred_label_idx]
print('real word is {}, predicted word is {}'.format(label, predict_word))

# 测试一下结果
word, label = trigram[19]
print('input: {}'.format(word))
print('label: {}'.format(label))
print()
word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
out = net(word)
pred_label_idx = out.max(1)[1].item()
predict_word = idx_to_word[pred_label_idx]
print('real word is {}, predicted word is {}'.format(label, predict_word))
