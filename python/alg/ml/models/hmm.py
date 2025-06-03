import numpy as np

class HMM(object):
    def __init__(self, N, M, pi=None, A=None, B=None):
        self.N = N
        self.M = M
        self.pi = pi
        self.A = A
        self.B = B

    def get_data_with_distribute(self, dist): # 根据给定的概率分布随机返回数据（索引）
        r = np.random.rand()
        for i, p in enumerate(dist):
            if r < p: return i
            r -= p

    def generate(self, T: int):
        '''
        根据给定的参数生成观测序列
        T: 指定要生成数据的数量
        '''
        z = self.get_data_with_distribute(self.pi)    # 根据初始概率分布生成第一个状态
        x = self.get_data_with_distribute(self.B[z])  # 生成第一个观测数据
        result = [x]
        for _ in range(T-1):        # 依次生成余下的状态和观测数据
            z = self.get_data_with_distribute(self.A[z])
            x = self.get_data_with_distribute(self.B[z])
            result.append(x)
        return result
    def evaluate(self, X):
        '''
        根据给定的参数计算条件概率
        X: 观测数据
        '''
        alpha = self.pi * self.B[:,X[0]]
        for x in X[1:]:
            # alpha_next = np.empty(self.N)
            # for j in range(self.N):
            #     alpha_next[j] = np.sum(self.A[:,j] * alpha * self.B[j,x])
            # alpha = alpha_next
            alpha = np.sum(self.A * alpha.reshape(-1,1) * self.B[:,x].reshape(1,-1), axis=0)
        return alpha.sum()
    def evaluate_backward(self, X):
        beta = np.ones(self.N)
        for x in X[:0:-1]:
            beta_next = np.empty(self.N)
            for i in range(self.N):
                beta_next[i] = np.sum(self.A[i,:] * self.B[:,x] * beta)
            beta = beta_next
        return np.sum(beta * self.pi * self.B[:,X[0]])

if __name__ == "__main__":
    pi = np.array([.25, .25, .25, .25])
    A = np.array([
        [0,  1,  0, 0],
        [.4, 0, .6, 0],
        [0, .4, 0, .6],
        [0, 0, .5, .5]])
    B = np.array([
        [.5, .5],
        [.3, .7],
        [.6, .4],
        [.8, .2]])
    hmm = HMM(4, 2, pi, A, B)
    print(hmm.generate(10))  # 生成10个数据
    print(hmm.evaluate([0,0,1,1,0]))   # 0.026862016