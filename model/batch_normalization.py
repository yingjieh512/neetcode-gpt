import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
         # 1. 将 Python List 转换成 NumPy 数组
        x = np.array(x, dtype=float)
        gamma = np.array(gamma, dtype=float)
        beta = np.array(beta, dtype=float)
        running_mean = np.array(running_mean, dtype=float)
        running_var = np.array(running_var, dtype=float)

        if training:
            # 2. 计算当前 Batch 每一列的均值
            mean = np.mean(x, axis=0)

            # 3. 计算当前 Batch 每一列的方差
            var = np.var(x, axis=0)

            # 4. 更新 running statistics
            # momentum 决定新 Batch 统计量所占的权重
            running_mean = (
                (1 - momentum) * running_mean
                + momentum * mean
            )

            running_var = (
                (1 - momentum) * running_var
                + momentum * var
            )

        else:
            # 5. 推理时使用训练期间保存的统计量
            mean = running_mean
            var = running_var

        # 6. 标准化
        x_hat = (x - mean) / np.sqrt(var + eps)

        # 7. 缩放和平移
        y = gamma * x_hat + beta

        # 8. 四舍五入并转换回 Python List
        return (
            np.round(y, 4).tolist(),
            np.round(running_mean, 4).tolist(),
            np.round(running_var, 4).tolist()
        )
