import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch



# convnext with bn
slak_tiny_300_knn = [72.378, 73.028, 73.532, 73.947, 74.432, 75.046, 75.346, 75.704, 75.71, 75.334, 75.128]
swin_tiny_300_knn = [70.398, 70.924, 71.248, 71.752, 72.276, 72.782, 73.550, 73.944, 74.022, 73.812, 73.752]
VIT_small_300_knn = [68.876, 69.322, 69.636, 70.004, 70.326, 71.182, 71.834, 72.492, 73.122, 73.438, 74.408]


plt.grid()
plt.rcParams['font.sans-serif'] = 'Times New Roman'

SIZE = 5
COLOR = '#3d405b'
MARKER = 'o'
# LABEL_FLAG = True
#for expert, para, perf, txt in zip(paras, paras, knn_perf_bn, txt_bn):
#     if LABEL_FLAG:
#         plt.scatter(para, perf, marker=MARKER, s=expert*SIZE, c=COLOR)
#         plt.annotate(txt, (para+0.05, perf+0.05), fontsize=15)
#     else:
#         plt.scatter(para, perf, marker=MARKER, s=expert*SIZE, c=COLOR)
#     # LABEL_FLAG = False
# plt.plot(paras, knn_perf_bn, c=COLOR, linestyle='dashdot', label='ConvSSL-T', linewidth=2)
#
#
# SIZE = 5
# COLOR = '#81b29a'
# MARKER = '^'
# LABEL_FLAG = True
# for expert, para, perf, txt_ in zip(paras_nobn, paras_nobn, knn_perf_nobn, txt_nobn):
#     if LABEL_FLAG:
#         plt.scatter(para, perf, marker=MARKER, s=expert*SIZE, c=COLOR)
#         plt.annotate(txt_, (para+0.05, perf+0.05), fontsize=15)
#     else:
#         plt.scatter(para, perf, marker=MARKER, s=expert*SIZE, c=COLOR)
#     # LABEL_FLAG = False
# plt.plot(paras_nobn, knn_perf_nobn, c=COLOR, linestyle='dashed', label='ConvNeXt-T', linewidth=2)
#
#
# plt.scatter(28.79, 69.118, marker='o', s=28.79*SIZE, c='purple', label='ConvNeXt-T (Rep)')
# plt.annotate('9x9', (28.79+0.05, 69.118+0.05), fontsize=15)
# # plt.scatter(32.1, 70.112, marker='o', s=32.1*SIZE, c='purple', label='ConvNeXt-T (RepLKNet)')
#
# # plt.scatter(23, 68.772, marker='o', s=30, c='blue', label='ViT-S')
#
# plt.scatter(28.3, 69.712, marker='o', s=28.3*SIZE, c='pink', label='Swin-T')
#
# # plt.xticks(paras, ['3x3', '5x5','7x7','9x9'])
# # plt.xlim((21, 49))
#
# plt.arrow(28.79, 70.324, 29.0-28.79, 0, head_width=0.00, head_length=0.00, linestyle=':' ,linewidth=3, color='red', length_includes_head=True)
# plt.arrow(28.53, 68.654, 29.0-28.53, 0, head_width=0.00, head_length=0.00, linestyle=':' ,linewidth=3, color='red', length_includes_head=True)
#
# plt.arrow(29.0, 68.654, 0, 70.324-68.654, head_width=0.02, head_length=0.03, linewidth=4, color='red', length_includes_head=True)
# plt.text(29.02, 69.25, '1.67', rotation=90, fontsize=18, color='red', weight='black')

plt.plot(range(len(slak_tiny_300_knn)), slak_tiny_300_knn, c='orange', linestyle='dashed', label='ConvSSL-T', linewidth=2)
plt.plot(range(len(swin_tiny_300_knn)), swin_tiny_300_knn, c='#81b29a', linestyle='dashed', label='Swin-T', linewidth=2)
plt.plot(range(len(VIT_small_300_knn)), VIT_small_300_knn, c=COLOR, linestyle='dashed', label='ViT-S', linewidth=2)
plt.xticks(range(len(swin_tiny_300_knn)), ['100', '120','140','160','180', '200','220','240','260', '280','300'])

FONTSIZE=15
plt.ylabel(r'K-NN Accuracy (%)', fontsize=FONTSIZE)
plt.xlabel(r'# Parameter Count (Millions)', fontsize=FONTSIZE)
plt.xticks(fontsize=FONTSIZE)
plt.yticks(fontsize=FONTSIZE)
plt.legend(fontsize=12)
plt.savefig('KNN_monitor.pdf', bbox_inches='tight')