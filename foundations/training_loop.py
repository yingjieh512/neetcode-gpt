import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        n, d = X.shape

        # 初始化模型参数
        # w: 每个特征对应一个权重，初始全为 0
        # b: 偏置，初始为 0
        w = np.zeros(d)
        b = 0.0

        # 训练 epochs 次
        for epoch in range(epochs):

            # 1. Forward：计算模型预测值
            # X: (n, d)
            # w: (d,)
            # y_hat: (n,)
            y_hat = X @ w + b

            # 2. 计算预测误差
            error = y_hat - y

            # 3. Backward：计算 MSE 对 w 的梯度
            # dw: (d,)
            dw = (2 / n) * (X.T @ error)

            # 4. 计算 MSE 对 b 的梯度
            db = (2 / n) * np.sum(error)

            # 5. Gradient Descent：更新参数
            w = w - lr * dw
            b = b - lr * db

        # 训练结束后返回参数
        return np.round(w, 5), round(b, 5)
