import os
import sys
import numpy as np
import matplotlib.pyplot as plt




dense_data = [
    [1.288E+17, 3.731E+10, 1.32857, 1],
    [1.572E+17, 4.000E+10, 1.28901, 2],
    [5.541E+17, 7.758E+10, 1.16041, 16],
    [4.548E+17, 6.818E+10, 1.21171, 12.5],
    [4.102E+17, 6.396E+10, 1.18986, 10.93],
    [3.556E+17, 5.879E+10, 1.17819, 9]]

offset = 2 * 3730636 - 4781260
expert = 4781260 - 3730636


# rmt = [1.6143, 1.2780, 1.1776, 1.1486]
# expert_rmt = [2,4,8,16]

# dense = [1.28901, 1.2361, 1.21171, 1.18986, 1.17819, 1.16041]


# # moe = [1.17089,1.17357,1.18149,1.18956] # top-2 smoe training
# # moe = [1.21059,1.18913,1.18494,1.18450] # top-12 smoe training
# moe = [1.20634,1.19076,1.18907,1.18982] # top-6 smoe training
# expert_moe = [2,4,8,16]


# convnext with bn
knn_perf_bn = [68.406, 68.798, 69.364, 70.324]  # 3 5 7 9 15
kernel_size = [27.5617, 27.6677, 27.8267, 28.0387]
paras = [27.5617, 27.6677, 27.8267, 28.0387]
txt_bn = ['3x3', '5x5', '7x7', '9x9' ]
# convnext withtou bn
kernel_size_nobn = [27.8267, 28.0387]
knn_perf_nobn = [68.654, 69.688] # 7 9 15
paras_nobn = [27.8267, 28.0387]
txt_nobn = ['7x7', '9x9' ]

# including 15x15
# convnext with bn
# knn_perf_bn = [68.406, 68.798, 69.364, 70.324, 69.728]  # 3 5 7 9 15
# kernel_size = [27.5617, 27.6677, 27.8267, 28.0387, 28.9926]
# paras = [27.5617, 27.6677, 27.8267, 28.0387, 28.9926]
#
# # convnext withtou bn
# kernel_size_nobn = [27.8267, 28.0387, 28.9926]
# knn_perf_nobn = [68.654, 69.688, 69.398] # 7 9 15
# paras_nobn = [27.8267, 28.0387, 28.9926]

plt.grid()
plt.rcParams['font.sans-serif'] = 'Times New Roman'

SIZE = 5
COLOR = '#3d405b'
MARKER = 'o'
LABEL_FLAG = True
for expert, para, perf, txt in zip(paras, paras, knn_perf_bn, txt_bn):
    if LABEL_FLAG:
        plt.scatter(para, perf, marker=MARKER, s=expert*SIZE, c=COLOR)
        plt.annotate(txt, (para+0.02, perf-0.1), fontsize=15)
    else:
        plt.scatter(para, perf, marker=MARKER, s=expert*SIZE, c=COLOR)
    # LABEL_FLAG = False
plt.plot(paras, knn_perf_bn, c=COLOR, linestyle='dashdot', label='ConvNeXt-SSL-T', linewidth=2)


SIZE = 5
COLOR = '#81b29a'
MARKER = '^'
LABEL_FLAG = True
for expert, para, perf, txt_ in zip(paras_nobn, paras_nobn, knn_perf_nobn, txt_nobn):
    if LABEL_FLAG:
        plt.scatter(para, perf, marker=MARKER, s=expert*SIZE, c=COLOR)
        plt.annotate(txt_, (para+0.02, perf-0.1), fontsize=15)
    else:
        plt.scatter(para, perf, marker=MARKER, s=expert*SIZE, c=COLOR)
    # LABEL_FLAG = False
plt.plot(paras_nobn, knn_perf_nobn, c=COLOR, linestyle='dashed', label='ConvNeXt-T', linewidth=2)


# plt.scatter(25, 58.884, marker='o', s=30, c='purple', label='ResNet-50')

# plt.scatter(23, 68.772, marker='o', s=30, c='blue', label='ViT-S')

plt.scatter(28, 69.712, marker='o', s=28*SIZE, c='pink', label='Swin-T')

# plt.xticks(paras, ['3x3', '5x5','7x7','9x9'])
# plt.xlim((21, 49))


FONTSIZE=15
plt.ylabel(r'K-NN Accuracy (%)', fontsize=FONTSIZE)
plt.xlabel(r'# Parameter Count (Millions)', fontsize=FONTSIZE)
plt.xticks(fontsize=FONTSIZE)
plt.yticks(fontsize=FONTSIZE)
plt.legend(fontsize=FONTSIZE)
plt.savefig('Teaser.pdf', bbox_inches='tight')