class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # x 是当前需要优化的参数，初始值为 init
        x = init

        # 执行指定次数的梯度下降
        for i in range(iterations):

            # f(x) = x^2 的导数是 2x
            # 导数表示当前位置函数增长最快的方向
            gradient = 2 * x

            # 沿着梯度的反方向移动，使函数值减小
            # learning_rate 控制每次移动的步长
            x = x - learning_rate * gradient

        # 返回最终的 x，保留 5 位小数
        return round(x, 5)