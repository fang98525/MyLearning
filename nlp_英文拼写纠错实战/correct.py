
import  numpy as  np

#读取词典库
vocab=set([line.rstrip() for line in open("./data/vocab.txt")])


#需要生成word所有的错误候选集合
def  geneaate_candiates(word):
    #生成编辑距离为1  单词
    letters= 'abcdefghijklmnopqrstuvwxyz'
    spilts=[[word[:i],word[i:]]  for i in range(len(word)+1)]
    #remove list
    remove=[L+R[1:]  for L,R in spilts if R]
    #添加list
    add=[L+c+R   for  c in letters  for L,R in spilts]
    #replace
    replace=[L+c+R[1:]  for c in letters for L,R in spilts if R]
    incorrect_list=add+replace+remove
    #用set 去重
    incorrect_list=set(incorrect_list)
    #过滤掉不在词典库的单词
    return  [word  for word in incorrect_list if word in vocab]
print(geneaate_candiates("apple"))   #['apples', 'ample', 'apply', 'apple']



#从nltk中加载语料库作为训练数据
# import nltk
# nltk.download('reuters')
# from nltk.corpus import reuters
# #读取语料库
# categories=reuters.categories()
# corpus=reuters.sents(categories=categories)
# print(categories,corpus)

#bigram 语言统计模型
#利用语料库构建语言模型  bigram模型  此处是源码实现  skLearn中也有封装好的包
corpus=["原始数据集划分成训练集和测试集以后","其中测试集除了用作调整参数","也用来测量模型的好坏","这样做导致最终的评分结果比实际效果要好"]
term_count = {}
bigram_count = {}
for doc in corpus:
    doc = ['<s>'] + list(doc)
    for i in range(0, len(doc) - 1):
        # bigram: [i,i+1]
        term = doc[i]
        bigram = doc[i:i + 2]

        if term in term_count:
            term_count[term] += 1
        else:
            term_count[term] = 1
        bigram = ' '.join(bigram)
        if bigram in bigram_count:
            bigram_count[bigram] += 1
        else:
            bigram_count[bigram] = 1
print(term_count,"\n",bigram_count)


# 一个单词被打错的概率    false  spell probability
prob={}

for line in open("./data/spell-errors.txt"):

    item=str(line).split(":")
    correct=item[0].strip()
    mistakes=[ word.strip() for word in item[1].strip().split(",")]
    # print(correct,mistakes)
    prob[correct]={}
    #每个错误写错成该错误的概率
    for  mistake in mistakes:
        prob[correct][mistake]=1.0/len(mistakes)
# print(prob)



#语言模型学习到的结果
print(term_count,"\n",bigram_count)
V = len(term_count.keys())   #所有token的个数
print(V)
file=open("./data/testdata.txt","r")

for line in file:
    # Python rstrip() 删除 string 字符串末尾的指定字符（默认为空格）.
    items = line.rstrip().split('\t')
    line = items[2].split()  # 得到每句话的单词列表
    # line = ["I", "like", "playing"]
    for word in line:
        if word not in vocab:
            # 需要替换word成正确的单词
            # Step1: 生成所有的(valid)候选集合
            # 对于没有在词典中出现过的拼写错误的单词 ，生成其编辑距离为1且正确的形式
            candidates = geneaate_candiates(word)

            # 如果没有 ，可以考虑生成编辑距离为2 的正确单词   此处不考虑这种情况
            if len(candidates) < 1:
                continue  # 不建议这么做（这是不对的）
            probs = []
            # 对于每一个candidate, 计算它的score
            # score = p(correct)*p(mistake|correct)
            #       = log p(correct) + log p(mistake|correct)
            # 返回score最大的candidate
            for candi in candidates:
                prob = 0
                # a. 计算channel probability
                if candi in prob and word in prob[candi]:
                    prob += np.log(prob[candi][word])
                else:
                    prob += np.log(0.0001)

                # b. 计算语言模型的概率
                idx = items[2].index(word) + 1
                if items[2][idx - 1] in bigram_count and candi in bigram_count[items[2][idx - 1]]:
                    prob += np.log((bigram_count[items[2][idx - 1]][candi] + 1.0) / (
                            term_count[bigram_count[items[2][idx - 1]]] + V))
                # TODO: 也要考虑当前 [word, post_word]
                #   prob += np.log(bigram概率)

                else:
                    prob += np.log(1.0 / V)

                probs.append(prob)

            max_idx = probs.index(max(probs))
            print(word, candidates[max_idx])