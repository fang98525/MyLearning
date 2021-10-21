"""
在图里面寻找最短路径的时候
在效率方面相对于粗暴地遍历所有路径，viterbi 维特比算法到达每一列的时候都会删除不符合最短路径要求的路径，大大降低时间复杂度。
对于每一轮的所有的节点 ，都找到其对于的最短路径 ，相当于剪枝了
"""
#使用维特比算法进行词性标注


#读取数据 建立词典   格式是每句话的每个词都有对应的词性

tag2id, id2tag = {}, {}  # maps tag to id . tag2id: {"VB": 0, "NNP":1,..} , id2tag: {0: "VB", 1: "NNP"....}
word2id, id2word = {}, {} # maps word to id
for line in open("./traindata.txt"):
    item=line.split("/")
    word,tag=item[0],item[1].rstrip()#去掉尾部空格
    if word  not in word2id:
        word2id[word]=len(id2word)
        id2word[len(id2word)] = word
    if tag not in  tag2id:
        tag2id[tag]=len(id2tag)
        id2tag[len(id2tag)] = tag
M = len(word2id)  # M: 词典的大小、# of words in dictionary
N = len(tag2id)   # N: 词性的种类个数  # of tags in tag set
# print(word2id)
# print(tag2id)
print(M,N)


#构建 pi：A，B    利用数据集统计得到模型的参数    基于概率统计的方法
"""解释（理解）：对于每个需要标注的句子，从第一个单词开始，
计算每个tag的概率，然后计算第二个的，选择到第二个概率最大的路径
相当于是有54个分支，路径长度为句子长度的求图的最短路径的问题
"""

#pi是在数据集上得到的每个tag的概率   注意理解统计的核心几行代码
import numpy as np
np.seterr(divide='ignore',invalid='ignore') #避免出现分母为0 情况
pi = np.zeros(N)   # 每个词性出现在句子中第一个位置的概率,  N: # of tags  pi[i]: tag i出现在句子中第一个位置的概率
A = np.zeros((N, M)) # A[i][j]: 给定tag i, 出现单词j的概率。 N: # of tags M: # of words in dictionary
B = np.zeros((N,N))  # B[i][j]: 之前的状态是i, 之后转换成转态j的概率 N: # of tags
pre_tga=""
for line in open("./traindata.txt"):
    items=line.split("/")
    # word1,tag1=item[0],item[1].rstrip()
    wordid,tagid=word2id[items[0]], tag2id[items[1].rstrip()]
    if pre_tga=="":#表示是句子的开始
        pi[tagid]+=1  #每个tag 在句子开头的概率
        A[tagid][wordid]+=1  #表示一个给定tag，出现单词wordid的概率
    else:  #表示不是开头
        A[tagid][wordid]+=1
        # 表示在前一个单词是tag1的情况下出现tagid的概率  或者理解为当前tag下，后一个单词是每个tag的概率
        B[tag2id[pre_tga]][tagid]+=1
    if items[0]=="." :#若到句子结尾  下一个为开头 oretag置为”“
        pre_tga=""
    else:#否则为当前词的tag
        pre_tga=items[1].rstrip()
print(pi)
#统计完成  计算概率
pi=pi/sum(pi)
for i in range(N):
    A[i]=np.true_divide(A[i],sum(A[i]))
    B[i] /= np.true_divide(B[i],sum(B[i])) #  到此为止计算完了模型的所有的参数： p, A, B  三个dp table
print(pi)

def  log(x):
    #这里是做了一个平滑处理 ，抵消出现为0 的情况
    if x==0:
        return  np.log(x+0.000001)
    else:return np.log(x)

# 根据模型 求句子的序列标注
def viterbi(x, pi, A, B):
    """
    x: user input string/sentence: x: "I like playing soccer"
    pi: initial probability of tags
    A: 给定tag, 每个单词出现的概率
    B: tag之间的转移概率
    """
    x = [word2id[word] for word in x.split(" ")]  # x: [4521, 412, 542 ..]
    T = len(x)

    dp = np.zeros((T, N))  # dp[i][j]: w1...wi, 假设wi的tag是第j个tag    每一步（某个状态n）有n（tag个数）个选择
    ptr = np.array([[0 for x in range(N)] for y in range(T)])  # T*N   dp table
    # TODO: ptr = np.zeros((T,N), dtype=int)

    for j in range(N):  # basecase for DP算法
        dp[0][j] = log(pi[j]) + log(A[j][x[0]])    #   初始化概率=第一个单词为某个tag的概率+ 在该tag下 出现句首单词的概率

    for i in range(1, T):  # 每个单词   句中的每一个单词
        for j in range(N):  # 每个词性  对句中的word 的每个tag
            # TODO: 以下几行代码可以写成一行（vectorize的操作， 会使得效率变高）
            dp[i][j] = -9999999
            for k in range(N):  # 从每一个k可以到达j
                score = dp[i - 1][k] + log(B[k][j]) + log(A[j][x[i]])
                if score > dp[i][j]:
                    dp[i][j] = score
                    ptr[i][j] = k

    # decoding: 把最好的tag sequence 打印出来
    best_seq = [0] * T  # best_seq = [1,5,2,23,4,...]
    # step1: 找出对应于最后一个单词的词性
    best_seq[T - 1] = np.argmax(dp[T - 1])

    # step2: 通过从后到前的循环来依次求出每个单词的词性
    for i in range(T - 2, -1, -1):  # T-2, T-1,... 1, 0
        best_seq[i] = ptr[i + 1][best_seq[i + 1]]

    # 到目前为止, best_seq存放了对应于x的 词性序列
    for i in range(len(best_seq)):
        print(id2tag[best_seq[i]])