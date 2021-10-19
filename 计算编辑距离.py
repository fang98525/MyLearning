"""
编辑距离的计算
编辑距离可以用来计算两个字符串的相似度，它的应用场景很多，其中之一是拼写纠正（spell correction）。
编辑距离的定义是给定两个字符串str1和str2, 我们要计算通过最少多少代价cost可以把str1转换成str2.
我们假定有三个不同的操作： 1. 插入新的字符 2. 替换字符 3. 删除一个字符。 每一个操作的代价为1.
"""

#基于dp的算法
def  edit_distance(str1,str2):
    #获取长度
    m,n=len(str1),len(str2)
    #生成dp  table
    dp=[[0 for x in range(n+1)] for x in range(m+1)]

#dp  填充数组
    for i in range(m+1):
        for j in range(n+1):
            if i==0:
                dp[i][j]=j
            if j==0:
                dp[i][j]=i
            elif str1[i-1]==str2[j-1]:  #若相同  不会产生cost
                dp[i][j]=dp[i-1][j-1]
            else:#三种选择里面最小的  花费加1
                dp[i][j]==1+min(dp[i][j-1],#相当于在前面的基础上添加
                                dp[i-1][j] ,#删除
                                dp[i-1][j-1]
                                )
        print(dp)
        return dp[i][j]
print(edit_distance("fang","jioo"))


#生成指定编辑距离的单词
#给定一个单词，我们也可以生成编辑距离为K的单词列表。
# 比如给定 str="apple"，K=1, 可以生成“appl”, "appla", "pple"...等 下面看怎么生成这些单词。 还是用英文的例子来说明。
# 仍然假设有三种操作 - 插入，删除，替换
def  generate_diatance_one(str1):
    letters="abcdefghijklmnopqrstuvwxyz"
    #对字符串进行分割
    spilt=[[str1[:i],str1[i:]] for i in range(len(str1)+1)]
    inserts=[L+c+R  for L,R in spilt   for  c in letters  if c] #循环相乘的情况
    remove=[L+R[1:] for L,R in spilt  if R]
    replace=[L+c+R[1:]  for  L,R in spilt if R for c in letters  if c]
     #去重
    # print(len(set(inserts+remove+replace)))
    return set(inserts+remove+replace)
print(generate_diatance_one("assf"))

#生成编辑距离为二的
str="appl"
data=[j  for i in generate_diatance_one(str)   for j in generate_diatance_one(i)]
print(len(data))



#基于jieba的分词
import  jieba
# 基于jieba的分词
seg_list = jieba.cut("贪心学院专注于人工智能教育", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))

jieba.add_word("贪心学院")
seg_list = jieba.cut("贪心学院专注于人工智能教育", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))



#判断一句话是否能够被字典切分   也是基于动态规划的解法
dic=set(["贪心科技", "人工智能", "教育", "在线", "专注于"])
print(dic)
def word_break(str):
    could_break=[False ]*(len(str)+1)
    could_break[0]=True
    for  i in range(1,len(could_break)):
        for  j in range(0,i):
            if str[j:i] in dic and could_break[j]==True:
                could_break[i]=True
    print(could_break)
    return  could_break[len(str)]
print(word_break("贪心科技在线教育"))

assert word_break("在线教育是")==False
assert word_break("")==True
assert word_break("在线教育人工智能")==True