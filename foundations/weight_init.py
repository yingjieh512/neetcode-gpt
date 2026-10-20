
import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # 固定随机种子
        torch.manual_seed(0)

        # Xavier Normal 初始化标准差
        std = math.sqrt(2 / (fan_in + fan_out))

        # 创建权重矩阵，形状为 (fan_out, fan_in)
        W = torch.randn(fan_out, fan_in) * std

        # 保留 4 位小数并转换为二维列表
        return W.round(decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # 固定随机种子
        torch.manual_seed(0)

        # Kaiming Normal 初始化标准差
        std = math.sqrt(2 / fan_in)

        # 创建权重矩阵，形状为 (fan_out, fan_in)
        W = torch.randn(fan_out, fan_in) * std

        # 保留 4 位小数并转换为二维列表
        return W.round(decimals=4).tolist()

    def check_activations(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        init_type: str
    ) -> List[float]:

        # 只在函数开始时固定一次随机种子
        torch.manual_seed(0)

        # ========== Step 1: 初始化所有权重 ==========

        weights = []
        fan_in = input_dim

        for _ in range(num_layers):

            # 当前层输出维度
            fan_out = hidden_dim

            # 选择初始化方法
            if init_type == "xavier":
                std = math.sqrt(2 / (fan_in + fan_out))

            elif init_type == "kaiming":
                std = math.sqrt(2 / fan_in)

            elif init_type == "random":
                std = 1.0

            else:
                raise ValueError("Invalid initialization type")

            # 创建当前层的权重矩阵
            W = torch.randn(fan_out, fan_in) * std

            # 保存权重
            weights.append(W)

            # 下一层的输入维度等于当前层输出维度
            fan_in = hidden_dim

        # ========== Step 2: 创建随机输入 ==========

        # Batch Size = 1
        # 输入形状为 (1, input_dim)
        x = torch.randn(1, input_dim)

        # ========== Step 3: Forward Propagation ==========

        stds = []

        for W in weights:

            # Linear Layer
            # x: (1, fan_in)
            # W.T: (fan_in, fan_out)
            z = x @ W.T

            # ReLU Activation
            x = torch.relu(z)

            # 记录当前层激活值的标准差
            stds.append(round(x.std().item(), 2))

        return stds
