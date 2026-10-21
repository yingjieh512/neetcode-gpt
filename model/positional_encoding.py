
import numpy as np
from numpy.typing import NDArray

class Solution:
    def get_positional_encoding(
        self,
        seq_len: int,
        d_model: int
    ) -> NDArray[np.float64]:

        # 第一步：创建位置索引
        # shape: (seq_len, 1)
        # 例如 seq_len=3 -> [[0], [1], [2]]
        positions = np.arange(seq_len)[:, np.newaxis]

        # 第二步：创建偶数维度索引
        # 例如 d_model=6 -> [0, 2, 4]
        dimensions = np.arange(0, d_model, 2)

        # 第三步：计算每个位置、维度对应的角度
        # Broadcasting:
        # (seq_len, 1) / (d_model//2,) 
        # -> (seq_len, ceil(d_model/2))
        angles = positions / (10000 ** (dimensions / d_model))

        # 第四步：创建位置编码矩阵
        # shape: (seq_len, d_model)
        PE = np.zeros((seq_len, d_model))

        # 偶数列使用 sin
        PE[:, 0::2] = np.sin(angles)

        # 奇数列使用 cos
        # 截取对应列，以兼容奇数 d_model
        PE[:, 1::2] = np.cos(angles[:, :d_model // 2])

        # 保留 5 位小数
        return np.round(PE, 5)
