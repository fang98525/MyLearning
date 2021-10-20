"""
文本分类是NLP最常见的应用。与大多数NLP应用一样，特征提取器Transformer模型近年来在该领域占据主导地位。
工具：Happy Transformer    构建在Hugging Face的transformers库
在Hugging Face的模型网络上，有100种预训练文本分类模型可供选择

比如 情感分析   检测金融数据的情绪
其他的可使用的模型版本：DistilBERT的普通版本作为起点。还有其他模型可以使用，如 BERT, ALBERT, RoBERTa等。更多型号请访问Hugging Face的模型网络：https://huggingface.co/models。

从0 搭建bert 训练部分吃透医学文本分类项目即可
 参考：https://mp.weixin.qq.com/s/X-WYv5AWEVt2x2LTk-Kcyg
"""

#hugging face transformer 库使用 Roberta   https://huggingface.co/models。
from transformers import RobertaTokenizer, RobertaModel
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
model = RobertaModel.from_pretrained('roberta-large')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)


#使用gpt  进行文本生成
from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)



#使用transformer  从0 开始训练 bert  也可以从预训练开始
"""
四个关键步骤：获取数据 、构建tokenizer、  创建输入管道、 训练model
OSCAR数据集是从互联网上获取的文本领域中最大的数据集之一。
OSCAR数据集拥有大量不同的语言，我们可以将BERT应用于一些不太常用的语言，如泰卢固语或纳瓦霍语。
"""

#tokenizer是流程中的关键组件。   对于不同的数据要调整不同的tokenizer  tokenizer对数据进行编码
#一般特殊token  包括 开始符  结束、未知、填充 掩码五种符号    用来编码文字
from transformers import RobertaTokenizer

# 初始化tokenizer
tokenizer = RobertaTokenizer.from_pretrained('filiberto', max_len=512)
# 在一个简单的句子上测试我们的tokenizer
tokens = tokenizer('ciao, come va?')
print(tokens)
{'input_ids': [0, 16834, 16, 488, 611, 35, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1]}
tokens.input_ids
[0, 16834, 16, 488, 611, 35, 2]


#输入管道的创建是更复杂的部分   包括 获取原始数据  对其进行转换，加载到打他loader中用来训练
#开始创建张量   使用MLM策略训练模型   需要 三种张量
import torch
batch=64
labels = torch.tensor([x.ids for x in batch])
mask = torch.tensor([x.attention_mask for x in batch])
# 复制标签张量，这将是input_ids
input_ids = labels.detach().clone()
# 创建与input_ids相同的随机浮点数数组
rand = torch.rand(input_ids.shape)
# 屏蔽15%的token
mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)
# 循环input_ids张量中的每一行(不能并行执行)
for i in range(input_ids.shape[0]):
    # 获取掩码位置的索引
    selection = torch.flatten(mask_arr[i].nonzero()).tolist()
    # 屏蔽
    input_ids[i, selection] = 3  # our custom [MASK] token == 3


#构建 dataloader
encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # 存储编码
        self.encodings = encodings

    def __len__(self):
        # 返回样本的数量
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, i):
        # 返回 input_ids, attention_mask 和 labels的字典
        return {key: tensor[i] for key, tensor in self.encodings.items()}


#利用transformer 库搭建roberta模型
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
config = RobertaConfig(
    vocab_size=30_522,  # 我们将其与标记器vocab_size对齐
    max_position_embeddings=514,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1
)
model = RobertaForMaskedLM(config)


#训练准备   设置GPU/CPU使用率。然后激活模型的训练模式，最后初始化优化器。定义loss等
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# 然后将我们的模型移到选定的设备上
model.to(device)
from transformers import AdamW
# 激活训练模式
model.train()
# 初始化优化器
optim = AdamW(model.parameters(), lr=1e-4)



#开始训练
import  tqdm
epochs = 2
for epoch in range(epochs):
    # 使用TQDM和dataloader设置循环
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # 初始化计算的梯度(从prev步骤)
        optim.zero_grad()
        # 训练所需的所有批
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # 处理
        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
        # 提取损失
        loss = outputs.loss
        # 计算每个需要更新的参数的损失
        loss.backward()
        # 更新参数
        optim.step()
        # 打印相关信息到进度条
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())




#保存模型
model.save_pretrained('./filiberto')

#然后评估等

