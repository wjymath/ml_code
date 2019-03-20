from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris


def create_data(n):
    np.random.seed(1)
    x_11 = np.random.randint(0, 100, (n, 1))
    x_12 = np.random.randint(0, 100, (n, 1))
    x_13 = 20 + np.random.randint(0, 100, (n, 1))
    x_21 = np.random.randint(0, 100, (n, 1))
    x_22 = np.random.randint(0, 100, (n, 1))
    x_23 = 10 - np.random.randint(0, 100, (n, 1))

    new_x_12 = x_12 * np.sqrt(2) / 2 - x_13 * np.sqrt(2) / 2
    new_x_13 = x_12 * np.sqrt(2) / 2 + x_13 * np.sqrt(2) / 2
    new_x_22 = x_22 * np.sqrt(2) / 2 - x_23 * np.sqrt(2) / 2
    new_x_23 = x_22 * np.sqrt(2) / 2 + x_23 * np.sqrt(2) / 2

    plus_samples = np.hstack([x_11, new_x_12, new_x_13, np.ones((n, 1))])
    mins_samples = np.hstack([x_21, new_x_22, new_x_23, -np.ones((n, 1))])

    samples = np.vstack([plus_samples, mins_samples])
    np.random.shuffle(samples)
    return samples


def plot_samples(ax, samples):
    Y = samples[:, -1]
    positon_p = Y == 1
    positon_n = Y == -1
    ax.scatter(samples[positon_p, 0], samples[positon_p, 1], samples[positon_p, 2], marker="+", label = "+", color = 'b')
    ax.scatter(samples[positon_n, 0], samples[positon_n, 1], samples[positon_n, 2], marker="^", label="-", color='y')


def perceptron(train_data, eta, w_0, b_0):
    x = train_data[:, :-1]
    y = train_data[:, -1]
    length = train_data.shape[0]
    w = w_0
    b = b_0
    step_num = 0
    while True:
        i = 0
        while(i < length):
            step_num += 1
            x_i = x[i].reshape((x.shape[1] ,1))
            y_i = y[i]
            if y_i * (np.dot(np.transpose(w), x_i) + b) < 0:
                w = w + eta * y_i * x_i
                b = b + eta * y_i
                break
            else:
                i += 1
        if(i == length):
            break
    return (w, b, step_num)


if __name__ == "__main__":
    data = create_data(100)
    eta = 0.1
    w_0 = np.ones((3,1), dtype=float)
    b_0 = 1
    w, b, num = perceptron(data, eta, w_0, b_0)
    print(w)
    print(b)
    fig = plt.figure()
    plt.suptitle("perceptron")
    ax = Axes3D(fig)
    plot_samples(ax, data)

    x = np.linspace(-30, 100, 100)
    y = np.linspace(-30, 100, 100)
    x, y = np.meshgrid(x, y)
    z = (-w[0][0] * x - w[1][0] * y - b) / w[2][0]
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='r',alpha=0.2)
    ax.legend(loc="best")
    plt.show()