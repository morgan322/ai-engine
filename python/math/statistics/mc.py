import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform

#目标采样分布的概率密度函数
def p(x):
    return 0.3*np.exp(-(x-0.3)**2)+0.7*np.exp(-(x-2.)**2/0.3)

#建议分布G：选择均值为1.4，方差为1.2的正态分布
G_rv = norm(loc=1.4, scale=1.2)

#常数值C=2.5
C=2.5

#均匀分布U(0,1)
uniform_rv = uniform(loc=0, scale=1)

#绘制目标分布的概率密度函数p(x)与建议分布G的概率密度函数g(x)的C倍
x = np.arange(-4., 6., 0.01)
plt.plot(x, p(x), color='r', lw=2, label='p(x)')
plt.plot(x, C*G_rv.pdf(x), color='b', lw=2, label='C*g(x)')  #g.pdf(x)表示正态分布的概率密度函数
plt.legend()
plt.show()

sample = []
#设10000个候选采样点
for i in range(10000):
    #step1：从建议分布G中进行采样，得到服从G分布的随机数
    Y = G_rv.rvs(1)[0]   #rvs():产生服从指定分布的随机数
    
    #step2：从均匀分布 U(0,1) 中进行采样，得到服从标准均匀分布的随机数
    U = uniform_rv.rvs(1)[0] 
    
    #step3：判断，如果 P(Y)≥U*C*g(Y)，则接受
    if p(Y)>=U*C*G_rv.pdf(Y):
        sample.append(Y)


#绘制目标分布的概率密度函数p(x)与建议分布G的概率密度函数g(x)的C倍
x = np.arange(-3., 5., 0.01)
plt.gca().axes.set_xlim(-3, 5)
plt.plot(x, p(x), color='r')
plt.hist(sample, color='b', bins=200, density=True, stacked=True, edgecolor='b')  #把所有直方归一化到1
plt.show()
