import numpy as np
import math
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#
def pixel_weight(height, width, sample, beta, bias):
    h = height
    w = width
    k = sample
    bias_sum = 0
    for l in range(bias):
        bias_sum += 1 / (l+1)
    w_weight = np.empty([h, w], dtype=float)
    h_weight = np.empty([h, w], dtype=float)
    # 宽度方向
    if w > 2 * k:
        alpha = (1/(1+bias) - beta/(k+bias)) / (beta - 1)
        for x in range(w):
            if x < k:
                weight = (1 / (x+1+bias) + alpha) / (2 * (math.log(k+bias)+0.5772-bias_sum) + (w - 2 * k)/(k + bias) + w * alpha)
                for i in range(h): w_weight[i][x] = weight
            elif x < (w-k):
                weight = (1 / (k + bias) + alpha) / (2 * (math.log(k+bias)+0.5772-bias_sum) + (w - 2 * k)/(k + bias) + w * alpha)
                for i in range(h): w_weight[i][x] = weight
            else:
                weight = (1 / (w-x+bias) + alpha) / (2 * (math.log(k+bias)+0.5772-bias_sum) + (w - 2 * k)/(k + bias) + w * alpha)
                for i in range(h): w_weight[i][x] = weight
    else:
        alpha = (1/(1+bias)-beta/(w-k+bias))/(beta -1)
        for x in range(w):
            if x < (w-k):
                weight = (1 / (x + 1 + bias) + alpha) / (2 * (math.log(w-k+bias)+0.5772-bias_sum) + (2 * k - w)/(w - k + bias) + w * alpha)
                for i in range(h): w_weight[i][x] = weight
            elif x < (2*k - w):
                weight = (1 / (w - k + bias) + alpha )/ (2 * (math.log(w-k+bias)+0.5772-bias_sum) + (2 * k - w)/(w - k + bias) + w * alpha)
                for i in range(h): w_weight[i][x] = weight
            else:
                weight = (1 / (w - x + bias) + alpha) / (2 * (math.log(w-k+bias)+0.5772-bias_sum) + (2 * k - w)/(w - k + bias) + w * alpha)
                for i in range(h): w_weight[i][x] = weight

    # 高度方向
    if h > 2 * k:
        alpha = (1/(1+bias) - beta/(k+bias)) / (beta - 1)
        for x in range(h):
            if x < k:
                weight = (1 / (x+1+bias) + alpha) / (2 * (math.log(k+bias)+0.5772-bias_sum) + (h - 2 * k)/(k + bias) + h * alpha)
                h_weight[x][:] = weight
            elif x < (h - k):
                weight = (1 / (k + bias) + alpha) / (2 * (math.log(k+bias)+0.5772-bias_sum) + (h - 2 * k)/(k + bias) + h * alpha)
                h_weight[x][:] = weight
            else:
                weight = (1 / (h-x+bias) + alpha) / (2 * (math.log(k+bias)+0.5772-bias_sum) + (h - 2 * k)/(k + bias) + h * alpha)
                h_weight[x][:] = weight
    else:
        alpha = (1/(1+bias)-beta/(h-k+bias))/(beta -1)
        for x in range(h):
            if x < (h - k):
                weight = (1 / (x + 1 + bias) + alpha) / (2 * (math.log(h-k+bias)+0.5772-bias_sum) + (2 * k - h)/(h - k + bias) + h * alpha)
                h_weight[x][:] = weight
            elif x < (2 * k - h):
                weight = (1 / (h - k + bias) + alpha )/ (2 * (math.log(h-k+bias)+0.5772-bias_sum) + (2 * k - h)/(h - k + bias) + h * alpha)
                h_weight[x][:] = weight
            else:
                weight = (1 / (h - x + bias) + alpha) / (2 * (math.log(h-k+bias)+0.5772-bias_sum) + (2 * k - h)/(h - k + bias) + h * alpha)
                h_weight[x][:] = weight

    # 总权重
    total_weight = (w_weight * h_weight) / np.sum(w_weight * h_weight)

    print(h_weight[:, 0])

    return total_weight
#
# def pixel_weight(height, width, sample, beta):
#     h = height
#     w = width
#     k = sample
#
#     w_weight = np.empty([h, w], dtype=float)
#     h_weight = np.empty([h, w], dtype=float)
#     # 宽度方向
#     if w > 2 * k:
#         alpha = 0
#         for x in range(w):
#             if x < k:
#                 weight = ((1 / (x+1)) + alpha) / (2 * (math.log(k)+0.5772) + (w - 2 * k)/k + w * alpha)
#                 for i in range(h): w_weight[i][x] = weight
#             elif x < (w-k):
#                 weight = (1 / k + alpha) / (2 * (math.log(k)+0.5772) + (w - 2 * k)/k + w * alpha)
#                 for i in range(h): w_weight[i][x] = weight
#             else:
#                 weight = (1 / (w-x) + alpha) / (2 * (math.log(k)+0.5772) + (w - 2 * k)/k + w * alpha)
#                 for i in range(h): w_weight[i][x] = weight
#     else:
#         alpha = 0
#         for x in range(w):
#             if x < (w-k):
#                 weight = ((1 / (x + 1)) + alpha) / (2 * (math.log(w-k)+0.5772) + (2 * k - w) / (w - k + 1) + w * alpha)
#                 for i in range(h): w_weight[i][x] = weight
#             elif x < (2*k - w):
#                 weight = ((1 / (w - k + 1)) + alpha) / (2 * (math.log(w-k)+0.5772) + (2 * k - w) / (w - k + 1) + w * alpha)
#                 for i in range(h): w_weight[i][x] = weight
#             else:
#                 weight = (1 / (w - x) + alpha) / (2 * (math.log(w-k)+0.5772) + (2 * k - w) / (w - k + 1) + w * alpha)
#                 for i in range(h): w_weight[i][x] = weight
#
#     # 高度方向
#     if h > 2 * k:
#         alpha = 0
#         for x in range(h):
#             if x < k:
#                 weight = ((1 / (x + 1)) + alpha) / (2 * (math.log(k)+0.5772) + (h - 2 * k) / k + h * alpha)
#                 h_weight[x][:] = weight
#             elif x < (h - k):
#                 weight = (1 / k + alpha) / (2 * (math.log(k)+0.5772) + (h - 2 * k) / k + h * alpha)
#                 h_weight[x][:] = weight
#             else:
#                 weight = (1 / (h - x) + alpha) / (2 * (math.log(k)+0.5772) + (h - 2 * k) / k + h * alpha)
#                 h_weight[x][:] = weight
#     else:
#         alpha = 0
#         for x in range(h):
#             if x < (h - k):
#                 weight = ((1 / (x + 1)) + alpha) / (2 * (math.log(h - k) + 0.5772) + (2 * k - h) / (h - k + 1) + h * alpha)
#                 h_weight[x][:] = weight
#             elif x < (2 * k - h):
#                 weight = ((1 / (h - k + 1)) + alpha) / (2 * (math.log(h - k) + 0.5772) + (2 * k - h) / (h - k + 1) + h * alpha)
#                 h_weight[x][:] = weight
#             else:
#                 weight = (1 / (h - x) + alpha) / (2 * (math.log(h - k) + 0.5772) + (2 * k - h) / (h - k + 1) + h * alpha)
#                 h_weight[x][:] = weight
#
#     # 总权重
#     # total_weight = (w_weight * h_weight) / np.sum(w_weight * h_weight)
#     total_weight = ((w_weight * h_weight)) / np.sum((w_weight * h_weight))
#     # print(total_weight)
#
#     return total_weight
#

weight = pixel_weight(512, 512, 200, 4, 20)

x,y = np.mgrid[1:512:512j, 1:512:512j]


fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x, y, weight, rstride=1, cstride=2, cmap=cm.coolwarm,alpha = 0.9)

ax.view_init(elev=20, azim=30)
plt.show()

# x = np.linspace(0, 620, 620)
# plt.plot(x, weight[:, 0])
# plt.show()