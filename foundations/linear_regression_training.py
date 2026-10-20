import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)
        N = len(X)

        # 复制初始权重，避免直接修改输入数组
        weights = initial_weights.copy()

        for _ in range(num_iterations):

            # 第一步：计算所有样本的预测值
            pred = self.get_model_prediction(X, weights)

            # 第二步：计算每个权重的梯度
            # 先存储所有梯度，再统一更新权重
            gradients = np.zeros_like(weights)

            for j in range(len(weights)):
                gradients[j] = self.get_derivative(
                    pred, Y, N, X, j
                )

            # 第三步：使用梯度下降更新所有权重
            weights -= self.learning_rate * gradients

        # 返回训练后的权重，保留 5 位小数
        return np.round(weights, 5)
