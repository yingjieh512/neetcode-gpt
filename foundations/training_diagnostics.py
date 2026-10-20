import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(
        self, model: nn.Module, x: torch.Tensor
    ) -> List[Dict[str, float]]:

        stats = []

        # 前向传播时不需要计算梯度
        with torch.no_grad():

            # 逐层遍历模型
            for layer in model.children():

                # 输入经过当前层，得到输出
                x = layer(x)

                # 只记录 Linear 层的统计数据
                if isinstance(layer, nn.Linear):

                    # 计算当前层输出的平均值
                    mean = x.mean().item()

                    # 计算当前层输出的标准差
                    std = x.std().item()

                    # 检查哪些神经元对整个 Batch
                    # 的输出都小于等于 0
                    if x.dim() >= 2:
                        dead = (x <= 0).all(dim=0)
                    else:
                        dead = (x <= 0)

                    # 死亡神经元占全部神经元的比例
                    dead_fraction = dead.float().mean().item()

                    stats.append({
                        "mean": round(mean, 4),
                        "std": round(std, 4),
                        "dead_fraction": round(dead_fraction, 4)
                    })

        return stats

    def compute_gradient_stats(
        self, model: nn.Module,
        x: torch.Tensor, y: torch.Tensor
    ) -> List[Dict[str, float]]:

        # 清除之前的梯度，防止梯度累积
        model.zero_grad()

        # Forward：计算预测结果
        prediction = model(x)

        # Loss：计算均方误差
        loss = nn.MSELoss()(prediction, y)

        # Backward：计算所有参数的梯度
        loss.backward()

        stats = []

        # 遍历模型中的 Linear 层
        for layer in model.children():

            if isinstance(layer, nn.Linear):

                # 获取当前 Linear 层权重的梯度
                grad = layer.weight.grad

                # 计算梯度平均值
                mean = grad.mean().item()

                # 计算梯度标准差
                std = grad.std().item()

                # 计算梯度的 L2 Norm
                norm = torch.norm(grad).item()

                stats.append({
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                    "norm": round(norm, 4)
                })

        return stats

    def diagnose(
        self,
        activation_stats: List[Dict[str, float]],
        gradient_stats: List[Dict[str, float]]
    ) -> str:

        # 优先级 1：死亡神经元
        for stat in activation_stats:
            if stat["dead_fraction"] > 0.5:
                return "dead_neurons"

        # 优先级 2：梯度爆炸
        for stat in gradient_stats:
            if stat["norm"] > 1000:
                return "exploding_gradients"

        # 优先级 3：梯度消失
        if gradient_stats:
            if gradient_stats[-1]["norm"] < 1e-5:
                return "vanishing_gradients"

        # 优先级 4：激活值异常
        for stat in activation_stats:

            if stat["std"] < 0.1:
                return "vanishing_gradients"

            if stat["std"] > 10.0:
                return "exploding_gradients"

        # 没有检测到上述异常
        return "healthy"