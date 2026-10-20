import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid

        # 第一层：784 个输入特征 -> 512 个隐藏神经元
        self.fc1 = nn.Linear(784, 512)

        # ReLU 激活函数：将负数变成 0
        self.relu = nn.ReLU()

        # Dropout：训练时随机将 20% 的元素置零
        self.dropout = nn.Dropout(0.2)

        # 第二层：512 个隐藏特征 -> 10 个输出
        self.fc2 = nn.Linear(512, 10)

        # Sigmoid：将输出映射到 0~1
        self.sigmoid = nn.Sigmoid()

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places

        # 1. 第一层线性变换
        x = self.fc1(images)

        # 2. ReLU 激活函数
        x = self.relu(x)

        # 3. Dropout 正则化
        x = self.dropout(x)

        # 4. 第二层线性变换
        x = self.fc2(x)

        # 5. Sigmoid 输出
        x = self.sigmoid(x)

        # 6. 保留四位小数
        return torch.round(x, decimals=4)