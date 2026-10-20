
import numpy as np
from numpy.typing import NDArray


class Solution:
    def lookup(
        self,
        embeddings: NDArray[np.float64],
        token_ids: NDArray[np.int64]
    ) -> NDArray[np.float64]:

        # embeddings 是二维矩阵
        # shape: (vocab_size, embed_dim)
        # 每一行存储一个 token 的向量

        # token_ids 是一维数组
        # 例如 [2, 0, 3]
        # 表示需要取出第 2、0、3 行

        # NumPy 高级索引，一次取出所有指定行
        result = embeddings[token_ids]

        # 按照题目要求保留 5 位小数
        return np.round(result, 5)
