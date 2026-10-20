import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64],
                weights: List[NDArray[np.float64]],
                biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:

        # 遍历每一层神经网络
        for i in range(len(weights)):

            # 线性变换：输入向量 @ 权重矩阵 + 偏置
            x = x @ weights[i] + biases[i]

            # 隐藏层使用 ReLU
            # 最后一层不使用激活函数
            if i < len(weights) - 1:
                x = np.maximum(0, x)

        return np.round(x, 5)