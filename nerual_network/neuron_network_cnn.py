# coding : utf8
import numpy as np

def conv2(X, k):
    x_row, x_col = X.shape  # 获取输入矩阵的行与列
    k_row, k_col = k.shape  # 获取卷积核的大小

    # padding
    ret_row, ret_col = x_row - k_row + 1, x_col - k_col + 1
    ret = np.empty((ret_row, ret_col))

    # convolution
    for y in range(ret_row):
        for x in range(ret_col):
            sub = X[y : y + k_row, x : x + k_col]
            ret[y, x] = np.sum(sub * k)
    return ret


def rot180(in_data):
    ret = in_data.copy()
    # 获取矩阵大小减一，目的是在翻转过程中防止翻转到矩阵外，使其在矩阵内翻转
    yEnd = ret.shape[0] - 1
    xEnd = ret.shape[1] - 1
    # 翻转代码，将一个矩阵分成四份，将左上右下换一个，左下右上换一个
    for y in range(ret.shape[0] / 2):
        for x in range(ret.shape[1]):
            ret[yEnd - y][x] = ret[y][x]
    for y in range(ret.shape[0]):
        for x in range(ret.shape[1] / 2):
            ret[y][xEnd - x] = ret[y][x]
    return ret


def padding(in_data, size):
    cur_r, cur_w = in_data.shape[0], in_data.shape[1]
    new_r = cur_r + size * 2
    new_w = cur_w + size * 2
    ret = np.zeros((new_r, new_w))
    ret[size : size + cur_r, size : size + cur_w] = in_data
    return ret


# one-hot
def discreterize(in_data, size):
    num = in_data.shape[0]
    ret = np.zeros((num, size))
    for i, idx in enumerate(in_data):
        ret[i, idx] = 1
    return ret


class ConvLayer:
    def __init__(self, in_channel, out_channel, kernel_size, lr = 0.01, momentum = 0.9, name = 'Conv'):
        # 随机初始化卷积核的参数
        self.w = np.random.randn(in_channel, out_channel, kernel_size, kernel_size)
        # 偏置初始化为0
        self.b = np.zeros((out_channel))
        self.layer_name = name
        self.lr = lr
        self.momentum = momentum

        self.prev_gradient_w = np.zeros_like(self.w)
        self.prev_gradient_b = np.zeros_like(self.b)

    # 前向传播
    def forward(self, in_data):
        print 'conv forward: ' + str(in_data.shape)
        # 获取batch_size，通道数，行数，列数
        in_batch, in_channel, in_row, in_col = in_data.shape
        # 获取卷积之后输出的维度，和卷积核的大小（一般行数列数均相同），输入的维度要保持和in_data的维度一致
        out_channel, kernel_size = self.w.shape[1], self.w.shape[2]
        self.top_val = np.zeros((in_batch, out_channel, in_row - kernel_size + 1, in_col - kernel_size + 1))
        self.bottom_val = in_data

        # 卷积
        for b_id in range(in_batch):
            for o in range(out_channel):
                for i in range(in_channel):
                    # 直接调用定义的卷积函数，对batch中的每一个输入矩阵卷积相加
                    self.top_val[b_id, o] += conv2(in_data[b_id, i], self.w[i, o])
                self.top_val[b_id, o] += self.b[o]
        return self.top_val

    def



