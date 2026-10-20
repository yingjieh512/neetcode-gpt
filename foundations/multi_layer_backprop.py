
import numpy as np
from typing import List


class Solution:
    def forward_and_backward(
        self, x, W1, b1, W2, b2, y_true
    ) -> dict:

        # 首先将所有输入转换为 NumPy 数组
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        # ========== Forward ==========

        # 第一层线性变换
        z1 = W1 @ x + b1

        # ReLU 激活函数
        a1 = np.maximum(0, z1)

        # 第二层线性变换
        y_pred = W2 @ a1 + b2

        # 计算 MSE
        loss = np.mean((y_pred - y_true) ** 2)

        # ========== Backward ==========

        # MSE 对预测值的导数
        dy = 2 * (y_pred - y_true) / y_true.size

        # 第二层的梯度
        dW2 = np.outer(dy, a1)
        db2 = dy

        # 把梯度传回第一层
        da1 = W2.T @ dy

        # ReLU 的导数
        dz1 = da1 * (z1 > 0)

        # 第一层的梯度
        dW1 = np.outer(dz1, x)
        db1 = dz1

        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist()
        }
