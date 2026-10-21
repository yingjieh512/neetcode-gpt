import torch
import torch.nn as nn
from torchtyping import TensorType


class TransformerBlock(nn.Module):

    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)

        # 1. 初始化多头自注意力模块
        self.attention = self.MultiHeadedSelfAttention(model_dim, num_heads)

        # 2. 初始化前馈神经网络
        self.feed_forward = self.VanillaNeuralNetwork(model_dim)

        # 3. 初始化两个 LayerNorm
        # Pre-LN：先归一化，再进入 Attention 或 FFN
        self.layer_norm_1 = nn.LayerNorm(model_dim)
        self.layer_norm_2 = nn.LayerNorm(model_dim)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)

        # 第一次残差连接
        # 先 LayerNorm，再 Attention，最后加上原始输入
        x = embedded + self.attention(self.layer_norm_1(embedded))

        # 第二次残差连接
        # 先 LayerNorm，再 Feed Forward，最后加上输入 x
        x = x + self.feed_forward(self.layer_norm_2(x))

        # 按题目要求，保留四位小数
        return torch.round(x, decimals=4)

    class MultiHeadedSelfAttention(nn.Module):

        class SingleHeadAttention(nn.Module):

            def __init__(self, model_dim: int, head_size: int):
                super().__init__()
                torch.manual_seed(0)

                # 将输入映射为 Key、Query、Value
                self.key_gen = nn.Linear(model_dim, head_size, bias=False)
                self.query_gen = nn.Linear(model_dim, head_size, bias=False)
                self.value_gen = nn.Linear(model_dim, head_size, bias=False)

            def forward(self, embedded: TensorType[float]) -> TensorType[float]:

                # 生成 K、Q、V
                # 形状：(batch_size, seq_len, head_size)
                k = self.key_gen(embedded)
                q = self.query_gen(embedded)
                v = self.value_gen(embedded)

                # 计算 QK^T，得到 Attention Scores
                # 形状：(batch_size, seq_len, seq_len)
                scores = q @ torch.transpose(k, 1, 2)

                context_length = k.shape[1]
                attention_dim = k.shape[2]

                # Scaled Dot-Product Attention
                # 除以 sqrt(head_size)，避免分数过大
                scores = scores / (attention_dim ** 0.5)

                # 创建下三角矩阵作为因果掩码
                # 让每个 token 只能关注自己及之前的 token
                lower_triangular = torch.tril(
                    torch.ones(
                        context_length,
                        context_length,
                        device=embedded.device
                    )
                )

                # 上三角位置设为 True，表示需要屏蔽
                mask = lower_triangular == 0

                # 将未来 token 的分数设为负无穷
                scores = scores.masked_fill(mask, float('-inf'))

                # Softmax 将分数转换为注意力权重
                scores = nn.functional.softmax(scores, dim=2)

                # 注意力权重乘以 V，生成每个 token 的新表示
                return scores @ v

        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            torch.manual_seed(0)

            # 每个 Head 处理的特征维度
            # 例如 model_dim=64, num_heads=4
            # 每个 Head 的 head_size=16
            assert model_dim % num_heads == 0

            self.att_heads = nn.ModuleList()

            # 创建多个独立的 Attention Head
            for i in range(num_heads):
                self.att_heads.append(
                    self.SingleHeadAttention(
                        model_dim,
                        model_dim // num_heads
                    )
                )

            # 多头拼接后，投影回 model_dim
            self.output_proj = nn.Linear(
                model_dim,
                model_dim,
                bias=False
            )

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:

            head_outputs = []

            # 依次计算每个 Attention Head 的输出
            for head in self.att_heads:
                head_outputs.append(head(embedded))

            # 沿最后一个维度拼接所有 Head
            # (batch, seq_len, head_size * num_heads)
            concatenated = torch.cat(head_outputs, dim=2)

            # 线性投影回 model_dim
            return self.output_proj(concatenated)

    class VanillaNeuralNetwork(nn.Module):

        def __init__(self, model_dim: int):
            super().__init__()
            torch.manual_seed(0)

            # 第一层：维度扩大 4 倍
            self.up_projection = nn.Linear(
                model_dim,
                model_dim * 4
            )

            # ReLU 激活函数
            self.relu = nn.ReLU()

            # 第二层：维度缩小回 model_dim
            self.down_projection = nn.Linear(
                model_dim * 4,
                model_dim
            )

            # Dropout：训练时随机将 20% 的输出置零
            self.dropout = nn.Dropout(0.2)

        def forward(self, x: TensorType[float]) -> TensorType[float]:
            torch.manual_seed(0)

            # Linear -> ReLU -> Linear -> Dropout
            return self.dropout(
                self.down_projection(
                    self.relu(
                        self.up_projection(x)
                    )
                ))