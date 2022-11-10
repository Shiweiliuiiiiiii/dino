import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch



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


# knn_perf_bn = [68.406, 68.798, 69.364, 70.324, 69.729]  # 3 5 7 9 15
# kernel_size = [28.33, 28.43, 28.53, 28.79, 29.76]
# paras =  [28.33, 28.43, 28.53, 28.79, 29.76]
# txt_bn = ['3x3', '5x5', '7x7', '9x9', '15x15']


# convnext with bn
knn_perf_bn = [68.406, 68.798, 69.364, 70.324, 69.729]  # 3 5 7 9 15
kernel_size = [28.33, 28.43, 28.53, 28.79, 29.76]
paras =  [28.33, 28.43, 28.53, 28.79, 29.76]
txt_bn = ['3x3', '5x5', '7x7', '9x9', '15x15']

# convnext without bn
kernel_size_nobn = [28.53, 28.79, 29.76]
knn_perf_nobn = [68.654, 69.688, 69.398] # 7 9 15
paras_nobn = [28.53, 28.79, 29.76 ]
txt_nobn = ['7x7', '9x9', '15x15']

plt.grid()
plt.rcParams['font.sans-serif'] = 'Times New Roman'

SIZE = 5
COLOR = '#3d405b'
MARKER = 'o'
LABEL_FLAG = True
for expert, para, perf, txt in zip(paras, paras, knn_perf_bn, txt_bn):
    if LABEL_FLAG:
        plt.scatter(para, perf, marker=MARKER, s=expert*SIZE, c='orange')
        plt.annotate(txt, (para+0.05, perf+0.05), fontsize=15)
    else:
        plt.scatter(para, perf, marker=MARKER, s=expert*SIZE, c='orange')
    # LABEL_FLAG = False
plt.plot(paras, knn_perf_bn, c='orange', linestyle='dashdot', label='BC-SSL-T', linewidth=2)


SIZE = 5
COLOR = '#81b29a'
MARKER = '^'
LABEL_FLAG = True
for expert, para, perf, txt_ in zip(paras_nobn, paras_nobn, knn_perf_nobn, txt_nobn):
    if LABEL_FLAG:
        plt.scatter(para, perf, marker=MARKER, s=expert*SIZE, c=COLOR)
        plt.annotate(txt_, (para+0.05, perf+0.05), fontsize=15)
    else:
        plt.scatter(para, perf, marker=MARKER, s=expert*SIZE, c=COLOR)
    # LABEL_FLAG = False
plt.plot(paras_nobn, knn_perf_nobn, c=COLOR, linestyle='dashed', label='ConvNeXt-T', linewidth=2)


# plt.scatter(28.79, 69.118, marker='o', s=28.79*SIZE, c='purple', label='ConvNeXt-T (Rep)')
# plt.annotate('9x9', (28.79+0.05, 69.118+0.05), fontsize=15)

# plt.scatter(32, 70.112, marker='o', s=28.79*SIZE, c='purple', label='ConvNeXt-T (Rep)')
# plt.annotate('31x31', (32+0.05, 70.112+0.05), fontsize=15)
# plt.scatter(32.1, 70.112, marker='o', s=32.1*SIZE, c='purple', label='ConvNeXt-T (RepLKNet)')

# plt.scatter(23, 68.772, marker='o', s=30, c='blue', label='ViT-S')

plt.scatter(28.3, 69.712, marker='o', s=28.3*SIZE, c='pink', label='Swin-T')

# plt.xticks(paras, ['3x3', '5x5','7x7','9x9'])
# plt.xlim((21, 49))

plt.arrow(28.79, 70.324, 29.0-28.79, 0, head_width=0.00, head_length=0.00, linestyle=':' ,linewidth=3, color='red', length_includes_head=True)
plt.arrow(28.53, 68.654, 29.0-28.53, 0, head_width=0.00, head_length=0.00, linestyle=':' ,linewidth=3, color='red', length_includes_head=True)

plt.arrow(29.0, 68.654, 0, 70.324-68.654, head_width=0.02, head_length=0.03, linewidth=4, color='red', length_includes_head=True)
plt.text(29.02, 69.25, '1.67', rotation=90, fontsize=18, color='red', weight='black')


FONTSIZE=15
plt.ylabel(r'K-NN Accuracy (%)', fontsize=FONTSIZE)
plt.xlabel(r'# Parameter Count (Millions)', fontsize=FONTSIZE)
plt.xticks(fontsize=FONTSIZE)
plt.yticks(fontsize=FONTSIZE)
plt.legend(fontsize=12)
plt.savefig('Teaser.pdf', bbox_inches='tight')