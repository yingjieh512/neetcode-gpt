
import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)

        # 创建 Embedding 层
        # vocabulary_size: 词表大小
        # 16: 每个 Token 对应 16 维向量
        self.embedding = nn.Embedding(vocabulary_size, 16)

        # 全连接层：16 维输入 -> 1 维输出
        self.linear = nn.Linear(16, 1)

        # 将输出转换为 0~1 之间的概率
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:

        # Embedding 层要求输入为整数 Token ID
        x = x.long()

        # (B, T) -> (B, T, 16)
        # 每个 Token ID 转换成 16 维向量
        x = self.embedding(x)

        # (B, T, 16) -> (B, 16)
        # 对句子中所有 Token 的向量求平均
        x = x.mean(dim=1)

        # (B, 16) -> (B, 1)
        # 将整个句子的向量映射成一个分数
        x = self.linear(x)

        # (B, 1) -> (B, 1)
        # 将分数转换为概率
        x = self.sigmoid(x)

        # 四舍五入到 4 位小数
        return torch.round(x, decimals=4)
