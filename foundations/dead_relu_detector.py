import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.

        # 保存每一个 ReLU 层的死亡神经元比例
        dead_fractions = []

        # 保存所有注册的 Hook，方便最后删除
        hooks = []

        # 定义 Hook：每次 ReLU 执行后自动调用
        def hook_fn(module, inputs, output):

            # output.shape = (batch_size, num_neurons)
            # 判断每个神经元是否对所有样本都输出 0
            dead = (output == 0).all(dim=0)

            # 计算死亡神经元占总神经元的比例
            fraction = dead.float().mean().item()

            # 保留 4 位小数
            dead_fractions.append(round(fraction, 4))

        # 遍历模型的所有层，包括嵌套的层
        for layer in model.modules():

            # 只给 ReLU 层注册 Hook
            if isinstance(layer, nn.ReLU):
                hooks.append(layer.register_forward_hook(hook_fn))

        try:
            # 不需要计算梯度
            with torch.no_grad():

                # 执行一次完整的 Forward Pass
                # ReLU 执行时会自动调用 hook_fn
                model(x)

        finally:
            # 删除 Hook，避免影响之后的模型运行
            for hook in hooks:
                hook.remove()

        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise

        # 情况 1：任意一层超过 50% 的神经元死亡
        if any(f > 0.5 for f in dead_fractions):
            return "use_leaky_relu"

        # 情况 2：第一层死亡比例超过 30%
        if dead_fractions and dead_fractions[0] > 0.3:
            return "reinitialize"

        # 情况 3：随着网络加深，死亡比例严格递增
        if len(dead_fractions) > 1:

            increasing = all(
                dead_fractions[i] < dead_fractions[i + 1]
                for i in range(len(dead_fractions) - 1)
            )

            # 同时要求最后一层死亡比例超过 10%
            if increasing and dead_fractions[-1] > 0.1:
                return "reduce_learning_rate"

        # 情况 4、5：没有达到以上异常阈值
        return "healthy"