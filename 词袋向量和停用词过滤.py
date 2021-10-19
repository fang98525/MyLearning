"""
出现频率特别高的和频率特别低的词对于文本分析帮助不大，一般在预处理阶段会过滤掉。 在英文里，经典的停用词为 “The”, "an"....
"""


#过滤
stop_words = ["the", "an", "is", "there"]
# 在使用时： 假设 word_list包含了文本里的单词
word_list = ["we", "are", "the", "students","play"]
filtered_words = [word for word in word_list if word not in stop_words]
print (filtered_words)

#nltk 进行词干提取
from nltk.stem.porter import *
stemmer = PorterStemmer()  #创建词干还原器

test_strs = ['caresses', 'flies', 'dies', 'mules', 'denied',
         'died', 'agreed', 'owned', 'humbled', 'sized',
         'meeting', 'stating', 'siezing', 'itemization',
         'sensational', 'traditional', 'reference', 'colonizer',
         'plotted']

singles = [stemmer.stem(word) for word in test_strs]
print(' '.join(singles))  # doctest: +NORMALIZE_WHITESPACE


# 词袋向量 的两种方法  根据词袋里面的单词进行embedding
#方法一  按照corpus里面词语出现的次数   基于出现次数的统计方法
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
corpus = [
     'He is going from Beijing to Shanghai.',
     'He denied my request, but he actually lied.',
     'Mike lost the phone, and phone was in the car.',
]
X = vectorizer.fit_transform(corpus)    #  fit  方法进行统计 训练概率数据
print (X.toarray())    #text的词袋向量
print (vectorizer.get_feature_names()) #向量每一维表示的单词

#方法二   词袋模型（tf-idf方法）
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(smooth_idf=False,norm="l2")  #smooth_idf 表示 平滑处理  平滑参数smooth_idf，在idf等于0的情况下的处理。 l2正则，需要标准化
X = vectorizer.fit_transform(corpus)