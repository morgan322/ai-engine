import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子确保结果可复现
np.random.seed(42)
torch.manual_seed(42)

class GDBTQuantitativeTrading:
    def __init__(self, stock_symbol, start_date, end_date, seq_length=60, n_trees=10, learning_rate=0.1):
        """初始化GDBT量化交易系统"""
        self.stock_symbol = stock_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.seq_length = seq_length
        self.n_trees = n_trees  # GDBT中的树数量
        self.learning_rate = learning_rate  # 学习率
        self.scaler = StandardScaler()
        self.models = []  # 存储GDBT中的每棵树
        self.losses = []  # 存储训练损失
        self.data = None
        self.train_data = None
        self.test_data = None
        self.predictions = None
        
    def fetch_data(self):
        """从Yahoo Finance获取股票数据"""
        try:
            self.data = yf.download(self.stock_symbol, start=self.start_date, end=self.end_date)
            print(f"成功获取{self.stock_symbol}的股票数据，共{len(self.data)}条记录")
            return self.data
        except Exception as e:
            print(f"获取数据时出错: {e}")
            return None
    
    def preprocess_data(self):
        """数据预处理与特征工程"""
        if self.data is None or len(self.data) == 0:
            print("没有数据可处理，请先获取数据")
            return False
        
        df = self.data.copy()
        
        # 计算技术指标
        # 移动平均线
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # 相对强弱指数(RSI)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 移动平均收敛发散(MACD)
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Histogram'] = df['MACD'] - df['Signal']
        
        # 布林带
        df['STD'] = df['Close'].rolling(window=20).std()
        df['Upper'] = df['MA20'] + (df['STD'] * 2)
        df['Lower'] = df['MA20'] - (df['STD'] * 2)
        
        # 动量
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        
        # 波动率
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # 移除包含NaN的行
        df.dropna(inplace=True)
        
        self.data = df
        print(f"数据预处理完成，剩余{len(self.data)}条记录")
        return True
    
    def prepare_sequences(self):
        """准备序列数据用于训练和测试"""
        if self.data is None or len(self.data) == 0:
            print("没有数据可处理，请先获取和预处理数据")
            return False
        
        # 提取特征和目标变量
        features = self.data.drop(['Close', 'Adj Close'], axis=1).values
        targets = self.data['Close'].values
        
        # 标准化特征
        scaled_features = self.scaler.fit_transform(features)
        
        # 创建序列数据
        X, y = [], []
        for i in range(self.seq_length, len(scaled_features)):
            X.append(scaled_features[i-self.seq_length:i])
            y.append(targets[i])  # 预测未来的收盘价
            
        X, y = np.array(X), np.array(y)
        
        # 划分训练集和测试集
        train_size = int(len(X) * 0.8)
        self.X_train, self.X_test = X[:train_size], X[train_size:]
        self.y_train, self.y_test = y[:train_size], y[train_size:]
        
        # 转换为PyTorch张量
        self.X_train_tensor = torch.FloatTensor(self.X_train)
        self.X_test_tensor = torch.FloatTensor(self.X_test)
        self.y_train_tensor = torch.FloatTensor(self.y_train).view(-1, 1)
        self.y_test_tensor = torch.FloatTensor(self.y_test).view(-1, 1)
        
        print(f"训练集大小: {len(self.X_train)}, 测试集大小: {len(self.X_test)}")
        return True
    
    def build_tree_model(self, input_size):
        """构建决策树模型（使用神经网络模拟）"""
        class TreeModel(nn.Module):
            def __init__(self, input_size):
                super(TreeModel, self).__init__()
                self.layer1 = nn.Linear(input_size, 128)
                self.layer2 = nn.Linear(128, 64)
                self.layer3 = nn.Linear(64, 32)
                self.output = nn.Linear(32, 1)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                x = x.view(x.size(0), -1)  # 展平输入
                x = self.relu(self.layer1(x))
                x = self.relu(self.layer2(x))
                x = self.relu(self.layer3(x))
                x = self.output(x)
                return x
                
        return TreeModel(input_size)
    
    def train_gdbt(self, epochs=100, batch_size=32, lr=0.001):
        """训练GDBT模型"""
        if not hasattr(self, 'X_train_tensor') or not hasattr(self, 'y_train_tensor'):
            print("请先准备序列数据")
            return False
        
        # 初始化第一个模型预测为训练数据的均值
        current_prediction = torch.full_like(self.y_train_tensor, self.y_train.mean())
        input_size = self.X_train_tensor.shape[1] * self.X_train_tensor.shape[2]
        
        # 训练多棵树
        for tree_idx in range(self.n_trees):
            print(f"训练第 {tree_idx+1}/{self.n_trees} 棵树")
            
            # 计算负梯度（残差）
            residuals = self.y_train_tensor - current_prediction
            
            # 创建数据加载器
            dataset = TensorDataset(self.X_train_tensor, residuals)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # 构建新树模型
            tree = self.build_tree_model(input_size)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(tree.parameters(), lr=lr)
            
            # 训练当前树
            tree_losses = []
            for epoch in range(epochs):
                epoch_loss = 0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = tree(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(dataloader)
                tree_losses.append(avg_loss)
                
                if (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
            self.losses.append(tree_losses)
            
            # 将当前树添加到模型集合
            self.models.append(tree)
            
            # 更新当前预测（使用学习率缩放）
            with torch.no_grad():
                tree_predictions = tree(self.X_train_tensor)
                current_prediction += self.learning_rate * tree_predictions
                
        print("GDBT模型训练完成")
        return True
    
    def predict(self):
        """使用GDBT模型进行预测"""
        if not self.models:
            print("请先训练GDBT模型")
            return None
        
        # 初始化预测结果
        test_predictions = torch.zeros_like(self.y_test_tensor)
        
        # 集成所有树的预测
        for tree in self.models:
            with torch.no_grad():
                tree_prediction = tree(self.X_test_tensor)
                test_predictions += self.learning_rate * tree_prediction
                
        self.predictions = test_predictions.numpy().flatten()
        self.actual_values = self.y_test_tensor.numpy().flatten()
        
        # 计算评估指标
        mse = mean_squared_error(self.actual_values, self.predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.actual_values, self.predictions)
        mape = np.mean(np.abs((self.actual_values - self.predictions) / self.actual_values)) * 100
        
        print(f"预测评估指标:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        # 绘制预测结果
        plt.figure(figsize=(12, 6))
        plt.plot(self.actual_values, label='实际价格')
        plt.plot(self.predictions, label='预测价格')
        plt.title(f'{self.stock_symbol} 股票价格预测')
        plt.xlabel('时间')
        plt.ylabel('价格')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'predictions': self.predictions,
            'actual_values': self.actual_values
        }
    
    def backtest_strategy(self):
        """回测交易策略"""
        if self.predictions is None:
            print("请先进行预测")
            return None
        
        # 获取测试期间的原始数据
        test_dates = self.data.index[-len(self.predictions):]
        test_close = self.data['Close'].values[-len(self.predictions):]
        
        # 创建交易决策DataFrame
        trading_df = pd.DataFrame({
            'Date': test_dates,
            'Actual_Close': test_close,
            'Predicted_Close': self.predictions
        })
        
        # 计算预测的价格变动方向
        trading_df['Predicted_Direction'] = np.where(
            trading_df['Predicted_Close'] > trading_df['Predicted_Close'].shift(1), 1, -1
        )
        
        # 计算实际价格变动和收益
        trading_df['Actual_Direction'] = np.where(
            trading_df['Actual_Close'] > trading_df['Actual_Close'].shift(1), 1, -1
        )
        trading_df['Market_Return'] = trading_df['Actual_Close'].pct_change()
        
        # 基于预测生成交易信号（简单策略：预测上涨则买入，预测下跌则卖出）
        trading_df['Signal'] = trading_df['Predicted_Direction'].shift(1)
        trading_df['Strategy_Return'] = trading_df['Signal'] * trading_df['Market_Return']
        
        # 计算累积收益
        trading_df['Cumulative_Market_Return'] = (1 + trading_df['Market_Return']).cumprod()
        trading_df['Cumulative_Strategy_Return'] = (1 + trading_df['Strategy_Return']).cumprod()
        
        # 计算评估指标
        total_days = len(trading_df)
        annual_market_return = (trading_df['Cumulative_Market_Return'].iloc[-1] ** (252 / total_days)) - 1
        annual_strategy_return = (trading_df['Cumulative_Strategy_Return'].iloc[-1] ** (252 / total_days)) - 1
        
        market_volatility = trading_df['Market_Return'].std() * np.sqrt(252)
        strategy_volatility = trading_df['Strategy_Return'].std() * np.sqrt(252)
        
        sharpe_market = annual_market_return / market_volatility if market_volatility != 0 else 0
        sharpe_strategy = annual_strategy_return / strategy_volatility if strategy_volatility != 0 else 0
        
        # 计算最大回撤
        trading_df['Market_Peak'] = trading_df['Cumulative_Market_Return'].cummax()
        trading_df['Strategy_Peak'] = trading_df['Cumulative_Strategy_Return'].cummax()
        trading_df['Market_Drawdown'] = trading_df['Cumulative_Market_Return'] / trading_df['Market_Peak'] - 1
        trading_df['Strategy_Drawdown'] = trading_df['Cumulative_Strategy_Return'] / trading_df['Strategy_Peak'] - 1
        
        max_market_drawdown = trading_df['Market_Drawdown'].min()
        max_strategy_drawdown = trading_df['Strategy_Drawdown'].min()
        
        # 胜率
        win_rate = len(trading_df[trading_df['Strategy_Return'] > 0]) / len(trading_df[trading_df['Strategy_Return'] != 0])
        
        print("\n==== 策略回测结果 ====")
        print(f"市场年化收益率: {annual_market_return:.2%}")
        print(f"策略年化收益率: {annual_strategy_return:.2%}")
        print(f"市场波动率: {market_volatility:.2%}")
        print(f"策略波动率: {strategy_volatility:.2%}")
        print(f"市场夏普比率: {sharpe_market:.2f}")
        print(f"策略夏普比率: {sharpe_strategy:.2f}")
        print(f"最大市场回撤: {max_market_drawdown:.2%}")
        print(f"最大策略回撤: {max_strategy_drawdown:.2%}")
        print(f"策略胜率: {win_rate:.2%}")
        
        # 绘制累积收益对比图
        plt.figure(figsize=(12, 6))
        plt.plot(trading_df['Date'], trading_df['Cumulative_Market_Return'], label='市场收益')
        plt.plot(trading_df['Date'], trading_df['Cumulative_Strategy_Return'], label='策略收益')
        plt.title(f'{self.stock_symbol} 市场与策略累积收益对比')
        plt.xlabel('日期')
        plt.ylabel('累积收益')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return {
            'trading_df': trading_df,
            'annual_market_return': annual_market_return,
            'annual_strategy_return': annual_strategy_return,
            'market_volatility': market_volatility,
            'strategy_volatility': strategy_volatility,
            'sharpe_market': sharpe_market,
            'sharpe_strategy': sharpe_strategy,
            'max_market_drawdown': max_market_drawdown,
            'max_strategy_drawdown': max_strategy_drawdown,
            'win_rate': win_rate
        }


# 主函数
def main():
    # 创建GDBT量化交易系统实例
    gdbt_trading = GDBTQuantitativeTrading(
        stock_symbol="AAPL",  # 股票代码
        start_date="2018-01-01",
        end_date="2023-07-01",
        seq_length=60,  # 序列长度
        n_trees=10,  # GDBT树的数量
        learning_rate=0.1  # 学习率
    )
    
    # 执行完整流程
    gdbt_trading.fetch_data()
    gdbt_trading.preprocess_data()
    gdbt_trading.prepare_sequences()
    gdbt_trading.train_gdbt(epochs=50, lr=0.001)
    prediction_results = gdbt_trading.predict()
    backtest_results = gdbt_trading.backtest_strategy()
    
    return gdbt_trading, prediction_results, backtest_results


if __name__ == "__main__":
    gdbt_trading, prediction_results, backtest_results = main()    