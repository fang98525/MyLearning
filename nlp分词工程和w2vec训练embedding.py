"""
实践内容包括文件的读取、中文分词、词向量表达、模型构建和模型融合。
语料库为复旦中文文本分类语料库，包含20个类别


"""

#读取一篇文本文档，去除stopwords，只保留中文字符后进行分词
import re
import os
import numpy as np
import pandas as pd
import jieba
import jieba.analyse


#打开文档  读取数据
def open_file(file_text):
    with open(file_text, "r",errors='ignore') as fp:
        content = fp.readlines()
    return content

#只保留中文字符
def remove(text):
    remove_chars = r'[^\u4e00-\u9fa5]'
    return re.sub(remove_chars, '', text)

#打开stopwords， 读取停用词表
def open_stop(file_stop):
    stopwords = [line.strip() for line in open(file_stop, 'r', encoding='utf-8').readlines()]
    return stopwords

#利用jieba对句子进行分词
def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    # stopwords = open_stop("stopwords.txt")
    stopwords=["模型","特点"]  #这些词会被剔除掉
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr.strip()
str="线性回归模型的特点是：最简单的回归模型、适用于几乎所有的回归问题、适用于大数据场景、可解释性比较好，应用场景非常多，比如：股价预测，房价预测，成绩预测，销量预测等"
print(seg_sentence((str)))


#从一个文本文件中提取中文分词，以句子或文章的形式保存在列表里
def extract_words_one_file(filepath,all=True):
    #all指的是是否以全文的形式保存所有分词。
    #True则表示一个文章的所有分词都储存在一个列表里，
    #False则表示每个句子的分词分别存在一个列表里，再以文章的形式储存列表

    inputs = open_file(filepath) #打开要加载的文件

    #获取每一句中的分词
    words_in_sentence = []
    for line in inputs:
        line_delete = remove(line)
        line_seg = seg_sentence(line_delete)  # 这里的返回值是字符串
        words_in_sentence.append(line_seg)

    words_in_sentence = [x for x in words_in_sentence if x != '']

    #利用空格切割获取所有分词
    alltokens = []
    chinesewords_sentence = []
    for i in range(len(words_in_sentence)):
        word = re.split(r'\s',words_in_sentence[i])
        alltokens.append(word)

    #对每一个句子产生的分词所储存的列表，删除空值
    for element in alltokens:
        element = [x for x in element if x != '']
        chinesewords_sentence.append(element)

    #获取所有分词的列表
    chinesewords_article = [i for k in chinesewords_sentence for i in k]

    if all == True:
        return chinesewords_article
    else:
        return chinesewords_sentence


"""
分词提取完毕后，我们需要用一定的方式表示这些分词，将这些计算机无法处理的非结构化信息转化为可计算的结构化信息。
one-hot方法是其中之一，。若文章或句子包含某词，则该词对应位置为1，否则为0。但是其在表示大规模文本的时候往往出现过于稀疏的现象，
也无法表示词语与词语间的关系，所以只是在比较简单的实践中使用。

本文我们所采取的文本表示方法是词嵌入中的Word2Vec。
通俗的来说，就是根据训练将分词用多维向量表示。
其2种训练模式为通过上下文来预测当前词和通过当前词来预测上下文。

"""

import gensim
#导入数据
with open("word_sentence.txt", "r") as f: #打开文件
    word_sentence = f.read() #读取文件
#eval() 函数用来执行一个字符串表达式，并返回表达式的值。
sent_feature = eval(word_sentence)
#我们只选取分词数大于3的句子进行W2v模型训练
sent_words = [i for k in sent_feature for i in k if len(i)>3]
#将分词的结果喂入模型
model = gensim.models.Word2Vec(sent_words, sg=1, size=100, window=3,iter=5,min_count=3, negative=3, sample=0.001, hs=1)
#模型保存
model.wv.save_word2vec_format('./word2vec_model.txt', binary=False)


"""在GitHub上看到了有已经训练好的中文词向量模型，个人认为这是一个比较好的资源，
毕竟感觉这些模型应该是依赖于很大的文本基数，充分利用可能也会提高最后的准确率。"""

#获取词向量后  ，便可以搭建各种模型，进行学习
