# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# 均值文件
mean_file='mean/image_mean.npy'
# mean_value: 104
# mean_value: 117
# mean_value: 123
mean=[104,117,123]
print(mean)

mean = np.load(mean_file).mean(1).mean(1)
print(mean)
