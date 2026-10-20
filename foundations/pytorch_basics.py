import torch
import torch.nn
from torchtyping import TensorType


class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        # 获取原始 Tensor 的总元素数量
        # numel() 返回 Tensor 中元素的总数
        total = to_reshape.numel()

        # 将 Tensor 变成 (总元素数 / 2, 2)
        result = torch.reshape(to_reshape, (total // 2, 2))

        return torch.round(result, decimals=4)

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        # dim=0 表示沿着行方向求平均
        # 相当于计算每一列的平均值
        result = torch.mean(to_avg, dim=0)

        return torch.round(result, decimals=4)

    def concatenate(self, cat_one: TensorType[float],
                    cat_two: TensorType[float]) -> TensorType[float]:
        # dim=1 表示沿着列方向拼接
        # 将两个 Tensor 左右连接起来
        result = torch.cat((cat_one, cat_two), dim=1)

        return torch.round(result, decimals=4)

    def get_loss(self, prediction: TensorType[float],
                 target: TensorType[float]) -> TensorType[float]:
        # MSE = Mean Squared Error（均方误差）
        # 计算预测值与真实值之间的平方误差平均值
        result = torch.nn.functional.mse_loss(prediction, target)

        return torch.round(result, decimals=4)