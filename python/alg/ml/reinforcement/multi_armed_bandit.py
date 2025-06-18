import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Callable

class MultiArmedBandit:
    """多臂老虎机环境"""
    def __init__(self, n_arms: int, means: List[float] = None, std_dev: float = 1.0):
        """
        初始化多臂老虎机环境
        
        参数:
            n_arms: 老虎机臂的数量
            means: 每个臂的真实奖励均值（默认随机生成）
            std_dev: 奖励的标准差
        """
        self.n_arms = n_arms
        self.std_dev = std_dev
        
        # 如果没有提供均值，则随机生成
        if means is None:
            self.means = np.random.normal(loc=0, scale=1, size=n_arms)
        else:
            self.means = np.array(means)
        
        # 最佳臂的索引和期望奖励
        self.best_arm = np.argmax(self.means)
        self.best_mean = np.max(self.means)
    
    def pull_arm(self, arm_index: int) -> float:
        """
        拉动指定索引的臂，返回随机奖励
        
        参数:
            arm_index: 臂的索引
        
        返回:
            奖励值（服从正态分布）
        """
        return np.random.normal(loc=self.means[arm_index], scale=self.std_dev)
    
    def get_regret(self, arm_index: int) -> float:
        """
        计算拉动指定臂的后悔值(regret)
        
        参数:
            arm_index: 臂的索引
        
        返回:
            后悔值 = 最佳臂的期望奖励 - 当前臂的期望奖励
        """
        return self.best_mean - self.means[arm_index]


class BanditAgent:
    """多臂老虎机智能体基类"""
    def __init__(self, n_arms: int):
        """
        初始化智能体
        
        参数:
            n_arms: 老虎机臂的数量
        """
        self.n_arms = n_arms
        self.action_counts = np.zeros(n_arms, dtype=int)  # 每个臂的选择次数
        self.action_values = np.zeros(n_arms)  # 每个臂的估计价值
    
    def select_action(self) -> int:
        """选择要拉动的臂"""
        raise NotImplementedError
    
    def update(self, arm_index: int, reward: float):
        """
        更新智能体对指定臂的估计
        
        参数:
            arm_index: 臂的索引
            reward: 获得的奖励
        """
        self.action_counts[arm_index] += 1
        
        # 使用增量式更新公式: Q(n+1) = Q(n) + (R-Q(n))/n
        step_size = 1.0 / self.action_counts[arm_index]
        self.action_values[arm_index] += step_size * (reward - self.action_values[arm_index])


class EpsilonGreedyAgent(BanditAgent):
    """ε-贪心策略智能体"""
    def __init__(self, n_arms: int, epsilon: float = 0.1):
        """
        初始化ε-贪心智能体
        
        参数:
            n_arms: 老虎机臂的数量
            epsilon: 探索概率
        """
        super().__init__(n_arms)
        self.epsilon = epsilon
    
    def select_action(self) -> int:
        """
        基于ε-贪心策略选择臂
        
        返回:
            选择的臂索引
        """
        if np.random.random() < self.epsilon:
            # 探索：随机选择一个臂
            return np.random.randint(0, self.n_arms)
        else:
            # 利用：选择当前估计价值最高的臂
            return np.argmax(self.action_values)


class UCBAgent(BanditAgent):
    """UCB(Upper Confidence Bound)策略智能体"""
    def __init__(self, n_arms: int, c: float = 2.0):
        """
        初始化UCB智能体
        
        参数:
            n_arms: 老虎机臂的数量
            c: 探索参数，控制置信区间的宽度
        """
        super().__init__(n_arms)
        self.c = c
        self.time_step = 0
    
    def select_action(self) -> int:
        """
        基于UCB策略选择臂
        
        返回:
            选择的臂索引
        """
        self.time_step += 1
        
        # 处理未被选择过的臂
        if np.any(self.action_counts == 0):
            return np.argmin(self.action_counts)
        
        # 计算UCB值: Q(a) + c * sqrt(ln(t)/N(a))
        ucb_values = self.action_values + self.c * np.sqrt(
            np.log(self.time_step) / self.action_counts
        )
        
        return np.argmax(ucb_values)


class ThompsonSamplingAgent(BanditAgent):
    """汤普森采样策略智能体（假设奖励服从正态分布）"""
    def __init__(self, n_arms: int):
        """
        初始化汤普森采样智能体
        
        参数:
            n_arms: 老虎机臂的数量
        """
        super().__init__(n_arms)
        # 初始化每个臂的后验分布参数（正态分布）
        self.mu = np.zeros(n_arms)  # 均值
        self.lambda_ = np.ones(n_arms)  # 精度(1/方差)
        self.alpha = np.ones(n_arms)  # 形状参数
        self.beta = np.ones(n_arms)  # 速率参数
    
    def select_action(self) -> int:
        """
        基于汤普森采样策略选择臂
        
        返回:
            选择的臂索引
        """
        # 从每个臂的后验分布中采样
        samples = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            tau = np.random.gamma(self.alpha[i], 1/self.beta[i])  # 采样精度
            mu = np.random.normal(self.mu[i], 1/(self.lambda_[i] * tau))  # 采样均值
            samples[i] = mu
        
        return np.argmax(samples)
    
    def update(self, arm_index: int, reward: float):
        """
        更新汤普森采样智能体的后验分布参数
        
        参数:
            arm_index: 臂的索引
            reward: 获得的奖励
        """
        # 更新参数
        lambda_old = self.lambda_[arm_index]
        mu_old = self.mu[arm_index]
        alpha_old = self.alpha[arm_index]
        beta_old = self.beta[arm_index]
        
        # 后验更新公式（正态-伽马共轭先验）
        self.lambda_[arm_index] = lambda_old + 1
        self.mu[arm_index] = (lambda_old * mu_old + reward) / self.lambda_[arm_index]
        self.alpha[arm_index] = alpha_old + 0.5
        self.beta[arm_index] = beta_old + 0.5 * lambda_old * (reward - mu_old)**2 / (lambda_old + 1)
        
        # 更新基础类中的动作计数和估计价值
        super().update(arm_index, reward)


def run_experiment(
    env: MultiArmedBandit, 
    agents: List[BanditAgent], 
    agent_names: List[str],
    num_steps: int,
    num_runs: int
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    运行多臂老虎机实验
    
    参数:
        env: 多臂老虎机环境
        agents: 智能体列表
        agent_names: 智能体名称列表
        num_steps: 每个实验的步数
        num_runs: 实验重复次数
    
    返回:
        包含每个智能体的平均奖励和累积后悔的字典
    """
    results = {}
    
    for agent, name in zip(agents, agent_names):
        # 初始化结果数组
        avg_rewards = np.zeros(num_steps)
        cumulative_regret = np.zeros(num_steps)
        
        # 重复多次实验取平均
        for _ in range(num_runs):
            # 重置环境和智能体
            agent.__init__(agent.n_arms)  # 重新初始化智能体
            
            rewards = np.zeros(num_steps)
            regrets = np.zeros(num_steps)
            
            for step in range(num_steps):
                # 选择动作
                action = agent.select_action()
                
                # 获取奖励
                reward = env.pull_arm(action)
                
                # 更新智能体
                agent.update(action, reward)
                
                # 记录奖励和后悔值
                rewards[step] = reward
                regrets[step] = env.get_regret(action)
            
            # 累积结果
            avg_rewards += rewards / num_runs
            cumulative_regret += np.cumsum(regrets) / num_runs
        
        results[name] = (avg_rewards, cumulative_regret)
    
    return results


def plot_results(results: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    """
    绘制实验结果
    
    参数:
        results: 包含每个智能体的平均奖励和累积后悔的字典
    """
    plt.figure(figsize=(14, 5))
    
    # 绘制平均奖励
    plt.subplot(1, 2, 1)
    for name, (avg_rewards, _) in results.items():
        plt.plot(avg_rewards, label=name)
    
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Average Reward over Time')
    plt.legend()
    plt.grid(True)
    
    # 绘制累积后悔
    plt.subplot(1, 2, 2)
    for name, (_, cumulative_regret) in results.items():
        plt.plot(cumulative_regret, label=name)
    
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Regret')
    plt.title('Cumulative Regret over Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


# 主函数：运行实验并展示结果
if __name__ == "__main__":
    # 设置随机种子以保证结果可复现
    np.random.seed(42)
    
    # 创建多臂老虎机环境
    n_arms = 10
    env = MultiArmedBandit(n_arms)
    
    # 创建不同策略的智能体
    epsilon_greedy = EpsilonGreedyAgent(n_arms, epsilon=0.1)
    ucb = UCBAgent(n_arms, c=2.0)
    thompson = ThompsonSamplingAgent(n_arms)
    
    # 运行实验
    results = run_experiment(
        env=env,
        agents=[epsilon_greedy, ucb, thompson],
        agent_names=["ε-Greedy (ε=0.1)", "UCB (c=2.0)", "Thompson Sampling"],
        num_steps=1000,
        num_runs=200
    )
    
    # 绘制结果
    plot_results(results)
    
    # 打印每个臂的真实均值和智能体估计值
    print("真实臂均值:", env.means)
    print("\nε-贪心估计值:", epsilon_greedy.action_values)
    print("UCB估计值:", ucb.action_values)
    print("汤普森采样估计值:", thompson.action_values)    