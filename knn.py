from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier

def generate_label(x):
    i = x
    if i%3==0 and i%5==0:
        # print("fizz buzz")
        return 0
    elif i%3 == 0:
        # print("fizz")
        return 1
    elif i%5 == 0:
        # print("buzz")
        return 2
    else:
        # print(i)
        return 3

# 2. 特征工程
# 将一个数据转换为多个特征
def feature_engine(num):
    feature = []
    feature.append(num%3)
    feature.append(num%5)
    feature.append(num%15)
    return feature

# 1. 数据准备
train_feature = []
train_label = []

test_feature = []
test_label = []

for num in range(100):
    train_feature.append(feature_engine(num))
    train_label.append(generate_label(num))

for num in range(100):
    num += 100
    test_feature.append(feature_engine(num))
    test_label.append(generate_label(num))

# print(train_feature[0])
# print(train_label[0])

# print(test_feature[0])
# print(test_label[0])

# 3. 训练模型
model = KNeighborsClassifier(n_neighbors=5)
model.fit(train_feature, train_label)
# 4. 评估模型
print(f"model train acc:{model.score(train_feature, train_label)}")

# 5. 预测模型
print(f"Sample:{test_feature[:2]}, Real:{test_label[:2]}, Pred:{model.predict(test_feature[:2])}")
# 6. 预测集评分
print(f"model test acc:{model.score(test_feature, test_label)}")

# 代码行数 56
# 准确度  1.0
# 是否需要准备数据 是
# 是否有可解释性 有
