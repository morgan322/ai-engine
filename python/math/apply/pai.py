# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import uniform
import seaborn
seaborn.set()

n = 100000
r = 1.0
o_x, o_y = (0., 0.)

uniform_x = uniform(o_x-r,2*r).rvs(n)
uniform_y = uniform(o_y-r,2*r).rvs(n)

d_array = np.sqrt((uniform_x - o_x) ** 2 + (uniform_y - o_y) ** 2)
res = sum(np.where(d_array < r, 1, 0))
pi = (res / n) /(r**2) * (2*r)**2

fig, ax = plt.subplots(1, 1)
ax.plot(uniform_x, uniform_y, 'ro', markersize=0.3)
plt.axis('equal')
circle = Circle(xy=(o_x, o_y), radius=r, alpha=0.5)
ax.add_patch(circle)

print('pi={}'.format(pi))
plt.show()


import random
total = [10, 100, 1000, 10000, 100000, 1000000, 5000000,10000000]  #随机点数
for t in total:
    in_count = 0
    for i in range(t):
        x = random.random()
        y = random.random()
        dis = (x**2 + y**2)**0.5
        if dis<=1:
            in_count += 1
    print(t,'个随机点时，π是：', 4*in_count/t)


#拉马努金计算圆周率
import math
#求阶乘的函数
def factorial(n):
   if n==0:
       return 1
   else:
       return n*factorial(n-1)

#计算π值的函数
def pi():
    sum =0
    k=0
    f=2*(math.sqrt(2))/9801
    while True:
        fz = (26390*k + 1103)*factorial(4*k)     #求和项分子
        fm = (396**(4*k))*((factorial(k))**4)    #求和项分母
        t = f*fz/fm
        sum += t
        if t<1e-15:                              #最后一项小于10^(-15)时跳出循环
            break
        k += 1                                   #更新k值
    return 1/sum

print("pi的值为：",pi())
#用于查看所写程序是否正确
print("pi的标准值为：",math.pi)


#Gauss–Legendre algorithm
#高斯-勒让德算法

a = 1
b = 1/math.sqrt(2)
t = 1/4
p = 1

w = 0

while w < 3:
    a1 = (a+b)/2
    b1 = math.sqrt(a*b)
    t1 = t-p*((a-(a+b)/2)**2)
    p1 = 2*p
    pi = ((a1+b1)**2)/(4*t1)

    a = a1
    b = b1
    t = t1
    p = p1

    w = w+1
    print(f'迭代{w}次计算圆周率：{pi}')




"""
Chudnovsky algorithm
"""
L = 13591409
X = 1
M = 1
K = 6
q = 0
sum = 0
while True:
    sum = sum + M*L/X
    pi = 426880*math.sqrt(10005)*(sum**-1)
    Lq = L + 545140134
    Xq = X * (-262537412640768000)
    Mq = M*((K**3-16*K)/(q+1)**3)
    Kq = K + 12
    q = q+1
    L = Lq
    X = Xq
    M = Mq
    K =Kq

    if pi == math.pi:
        break

print(f'迭代{q}次，圆周率：{pi}')


# import math
# from decimal import Decimal, getcontext
# import sys
# import time

# # 设置Decimal的精度,这里设置得很高以便于长时间运行.
# getcontext().prec = 1000

# def compute_pi():
#     """无限计算π的近似值,并实时在同一行更新显示结果"""
#     C = Decimal(426880 * math.sqrt(10005))
#     M = 1
#     L = 13591409
#     X = 1
#     K = 6
#     S = Decimal(L)
#     i = 0
#     while True:
#         M = (K**3 - 16*K) * M // (i+1)**3 
#         L += 545140134
#         X *= -262537412640768000
#         S += Decimal(M * L) / X
#         K += 12
#         pi_approx = C / S
#         sys.stdout.write("\r" + str(pi_approx)[:2 + i])  # 逐步显示更多位的π
#         sys.stdout.flush()
#         time.sleep(0.1)  # 减慢输出速度
#         i += 1

# # 开始计算π的值, 无限输出
# compute_pi()