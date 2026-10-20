import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        z=np.dot(x,w)+b
        y_hat=1/(1+np.exp(-z))
        loss=0.5*(y_hat-y_true)
        # Sigmoid 的导数
        dy_dz = y_hat * (1 - y_hat)
        dL_dy=y_hat-y_true

# Loss 对 Logit z 的导数
        dL_dz = dL_dy * dy_dz

# Loss 对权重和偏置的导数
        dL_dw = dL_dz * x
        dL_db = dL_dz
        return np.round(dL_dw, 5), round(float(dL_db), 5)
