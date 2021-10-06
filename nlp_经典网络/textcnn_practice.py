
import  numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
'''
使用textcnn进行情感分析的简易模板   
'''
embedding_size=2
sequence_length=3
num_classes=2
filter_sizes=[2,2,2]
num_filters=3
sentence=["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
labels=[1,1,1,0,0,0]
#生成词典，作为数字化表示
word_list=" ".join(sentence).split()
word_list=list(set(word_list))
dict={w:i  for i,w in enumerate(word_list)}
print(dict)
vocab_size=len(dict)

#特征化
inputs=torch.LongTensor([np.array([dict[n] for  n in sen.split()]) for sen in sentence])
ladels=torch.LongTensor([label for label in labels])

#定义模型
class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.num_filters_total = num_filters * len(filter_sizes)
        self.W = nn.Embedding(vocab_size, embedding_size)
        self.Weight = nn.Linear(self.num_filters_total, num_classes, bias=False)
        self.Bias = nn.Parameter(torch.ones([num_classes]))
        self.filter_list = nn.ModuleList([nn.Conv2d(1, num_filters, (size, embedding_size)) for size in filter_sizes])

    def forward(self, X):
        embedded_chars = self.W(X)  # [batch_size, sequence_length, sequence_length]
        embedded_chars = embedded_chars.unsqueeze(
            1)  # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]

        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            # conv : [input_channel(=1), output_channel(=3), (filter_height, filter_width), bias_option]
            h = F.relu(conv(embedded_chars))
            # mp : ((filter_height, filter_width))
            mp = nn.MaxPool2d((sequence_length - filter_sizes[i] + 1, 1))
            # pooled : [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3)]
            pooled = mp(h).permute(0, 3, 2, 1)
            pooled_outputs.append(pooled)

        h_pool = torch.cat(pooled_outputs, len(
            filter_sizes))  # [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3) * 3]
        h_pool_flat = torch.reshape(h_pool, [-1,
                                             self.num_filters_total])  # [batch_size(=6), output_height * output_width * (output_channel * 3)]
        model = self.Weight(h_pool_flat) + self.Bias  # [batch_size, num_classes]
        return model



#训练
model=TextCNN()
print(model)
loss=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.0001)

for epoch  in range(5000):
    optimizer.zero_grad()
    output=model(inputs)
    res_loss=loss(output,ladels)
    if (epoch+1)%500==0:
        #输出结果用到了两种范式
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(res_loss))
    res_loss.backward()
    optimizer.step()

#predict
print(model.parameters())

test_text = 'sorry loves you'
tests = [np.asarray([dict[n] for n in test_text.split()])]
test_batch = torch.LongTensor(tests)

prediction=model(test_batch).data.max(1,keepdim=True)[1]
print(prediction)
if prediction[0][0] == 0:
    print(test_text, "is Bad Mean...")
else:
    print(test_text, "is Good Mean!!")
