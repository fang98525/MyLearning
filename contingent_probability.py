import matplotlib.pyplot as plt
import numpy as np

#协方差公式  计算两个变量的相关性
def Cov(x1, x2):
    """
    协方差，衡量2个变量之间的相关性
    """
    return np.mean((x1-np.mean(x1))*(x2-np.mean(x2)))

def rho(x1, x2):
    """
    归一化，使2个变量的相关性收敛到0~1之间衡量
    """
    return Cov(x1,x2)/(np.std(x1)*np.std(x2))

def N(x, mu, xstd):
    return 1/(np.sqrt(2*np.pi)*xstd)*np.exp(-(x-mu)**2/(2*xstd**2))

def factorial(a):
    if a == 0:
        return 1
    else:
        return a*factorial(a-1)

def factorial2(f, n, m):
    print("n:{0},(f-m+1):{1}".format(n, (f-m+1)))
    if n == (f-m+1):
        return (f-m+1)
    else:
        return n*factorial2(f, n-1, m)

def B(n, k, p):
    return factorial2(n, n, k)/factorial(k)*p**k*(1-p)**(n-k)

# x1 = np.random.normal(1.0, 1.0, 100)
# x2 = np.random.normal(0, 2.0, 100)

x1 = np.random.binomial(1, 0.3, 100)
x2 = np.random.binomial(1, 0.6, 100)
# print(x1)

# px1 = len(x1[x1>2])/len(x1)
# px2 = len(x2[x2>2])/len(x2)

# p = px1*px2

plt.hist(x1, density=True, bins=2)
plt.hist(x2, density=True, bins=2)

# 创建等差数列
# xplt = np.linspace(-6, 6, 100)
# print(xplt)

# # 此处特别注意，输入的xplt是新定义的横轴，mu、std都是输入的随机变量x1,x2
# xplt1 = N(xplt, np.mean(x1), np.std(x1))
# xplt2 = N(xplt, np.mean(x2), np.std(x2))

# xplt1 = B(len(x1), len(x1[x1==1]), len(x1[x1==1])/len(x1))
# xplt2 = B(len(x2), len(x1[x2==1]), len(x2[x2==1])/len(x2))

# # # # print(f"cov(x1,x2):{Cov(x1,x2)}, p1:{px1}, p2:{px2}, p:{p},rho:{rho(x1, x2)}")
# print(f"cov(x1,x2):{Cov(x1,x2)} ,rho:{rho(x1, x2)}")
# print(f"xplt1:{xplt1} ,xplt2:{xplt2}")

# plt.scatter(x1, x2)
# plt.scatter(xplt1, xplt2)
# plt.plot(xplt, xplt1, color="r")
# plt.plot(xplt, xplt2, color="g")

plt.show()
