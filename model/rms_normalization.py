import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:

        # 1. 把输入转换成 NumPy 数组
        x = np.array(x, dtype=float)
        gamma = np.array(gamma, dtype=float)

        # 2. 计算所有特征的平方均值 Mean Square
        mean_square = np.mean(x ** 2)

        # 3. 计算 RMS（均方根）
        # eps 防止分母为 0
        rms = np.sqrt(mean_square + eps)

        # 4. 用 RMS 归一化每个特征
        x_hat = x / rms

        # 5. 使用 gamma 缩放
        output = gamma * x_hat

        # 6. 保留四位小数，转换成 Python List
        return np.round(output, 4).tolist()