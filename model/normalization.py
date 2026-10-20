import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(
        self,
        x: NDArray[np.float64],
        gamma: NDArray[np.float64],
        beta: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        # 1. 计算输入向量的平均值
        mean = np.mean(x)

        # 2. 计算输入向量的方差
        var = np.var(x)

        # 3. 添加 eps，防止除以 0
        eps = 1e-5

        # 4. 标准化：减去均值，再除以标准差
        x_hat = (x - mean) / np.sqrt(var + eps)

        # 5. 使用 gamma 缩放，beta 平移
        out = gamma * x_hat + beta

        # 6. 保留 5 位小数
        return np.round(out, 5)