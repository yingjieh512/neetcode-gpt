from typing import List
from collections import Counter

class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        tokens = list(corpus)
        result = []

        for _ in range(num_merges):
            # 1. 统计相邻 pair 的出现次数
            pairs = Counter()

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] += 1

            if not pairs:
                break

            # 2. 找到最高频 pair
            # 如果频率相同，选字典序最小的
            best = min(pairs, key=lambda x: (-pairs[x], x))

            result.append(list(best))

            # 3. 合并所有不重叠的 pair
            new_tokens = []
            i = 0

            while i < len(tokens):
                if (i + 1 < len(tokens)
                    and (tokens[i], tokens[i + 1]) == best):

                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return result