import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

# from mpl_toolkits.mplot3d import Axes3D

def dot(v, w):
    prod = [v_i * w_i for v_i, w_i in zip(v, w)]
    return sum(prod)


def mean(v):
    return (sum(v) / len(v))


def de_mean(v):
    v_bar = mean(v)
    return [v_i - v_bar for v_i in v]


def sum_of_squares(v):
    return dot(v, v)


def variance(v):
    n = len(v)
    dist_from_mean = de_mean(v)
    sum_sq_diff = sum_of_squares(dist_from_mean)
    return (sum_sq_diff / (n - 1))


def standard_deviation(v):
    return math.sqrt(variance(v))


def covariance(v, w):
    n = len(v)
    return dot(de_mean(v), de_mean(w)) / (n - 1)


def correlation(v, w):
    std_v = standard_deviation(v)
    std_w = standard_deviation(w)
    return covariance(v, w) / (std_v * std_w)


def correlation_coefficient(rx1y, rx2y, rx1x2):
    return (((rx1y ** 2 + rx2y ** 2) - (2 * rx1y * rx2y * rx1x2)) / (1 - (rx1x2 ** 2))) ** 1 / 2


def beta_1(X1, X2, Y):
    rx1y = correlation(X1, Y)
    rx2y = correlation(X2, Y)
    rx1x2 = correlation(X1, X2)
    return ((rx1y - (rx2y * rx1x2)) / (1 - (rx1x2) ** 2)) * (standard_deviation(Y) / standard_deviation(X1))


def beta_2(X1, X2, Y):
    rx1y = correlation(X1, Y)
    rx2y = correlation(X2, Y)
    rx1x2 = correlation(X1, X2)
    return ((rx2y - (rx1y * rx1x2)) / (1 - (rx1x2) ** 2)) * (standard_deviation(Y) / standard_deviation(X2))


def alpha(X1, X2, Y):
    rx1y = correlation(X1, Y)
    rx2y = correlation(X2, Y)
    rx1x2 = correlation(X1, X2)
    b_1 = beta_1(X1, X2, Y)
    b_2 = beta_2(X1, X2, Y)
    return (mean(Y) - (b_1 * mean(X1)) - ((b_2 * mean(X2))))


def Multiple_Regression(X1, X2, Y):
    rx1y = correlation(X1, X2)
    rx2y = correlation(X1, Y)
    rx1x2 = correlation(X2, Y)

    Beta_1 = beta_1(X1, X2, Y)
    Beta_2 = beta_2(X1, X2, Y)
    Alpha = alpha(X1, X2, Y)

    return Beta_1, Beta_2, Alpha


if __name__ == '__main__':
    X_1 = [12, 14, 15, 16, 18]
    X_2 = [32, 35, 45, 50, 65]
    Y_ = [350000, 399765, 429000, 435000, 433000]
    [B1, B2, Alpha] = Multiple_Regression(X_1, X_2, Y_)
    Predicted_Value = [(x_i * B1) + (x_j * B2) + Alpha for x_i, x_j in zip(X_1, X_2)]
    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)
    ax.scatter(X_1, X_2, Y_)
    ax.plot(X_1, X_2, Predicted_Value, '--')
    xLabel = ax.set_xlabel("Highest Year of School Completed")
    yLabel = ax.set_ylabel("Motivation by Huggins Motivation Scale")
    zLabel = ax.set_zlabel("Predicted Annual Sales")
    plt.show()
    print("Beta_1: %f" % B1)
    print("Beta_2: %f" % B2)
    print("Alpha: %f" % Alpha)