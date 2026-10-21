import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
                # 第一步：合并所有正面和负面句子
        sentences = positive + negative

        # 第二步：收集所有不重复的单词
        vocab = set()

        for sentence in sentences:
            # split() 按空白字符切分句子
            words = sentence.split()

            for word in words:
                vocab.add(word)

        # 第三步：按字母顺序排序，分配 Token ID
        # ID 从 1 开始，因为 0 留给 Padding
        vocab = sorted(vocab)

        word_to_id = {}

        for i, word in enumerate(vocab):
            word_to_id[word] = i + 1

        # 第四步：将每个句子编码成 Token ID
        tensors = []

        for sentence in sentences:
            words = sentence.split()

            # 根据词表，把单词转换为对应的数字
            ids = [word_to_id[word] for word in words]

            # Token ID 应使用整数类型
            tensors.append(torch.tensor(ids, dtype=torch.long))

        # 第五步：Padding
        # 把短句子补 0，使所有句子长度相同
        result = nn.utils.rnn.pad_sequence(
            tensors,
            batch_first=True,
            padding_value=0
        )

        return result