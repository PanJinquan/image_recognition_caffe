# -*-coding: utf-8 -*-
'''
  在Caffe中,彩色图像的通道要求是BGR格式，输入数据是float32类型,范围[0,255]
  [1]caffe的训练/测试prototxt文件,一般在数据层设置:cale:0.00392156885937,即1/255.0,即将数据归一化到[0,1]
  [2]当输入数据为RGB图像,float32,[0,1],则需要转换:
    --transformer.set_raw_scale('data',255)       # 缩放至0~255
    --transformer.set_channel_swap('data',(2,1,0))# 将RGB变换到BGR
  [3]当输入数据是RGB图像,uint8类型,[0,255],则输入数据之前必须乘以*1.0转换为float32
    --transformer.set_raw_scale('data',1.0)       # 数据不用缩放了
    --transformer.set_channel_swap('data',(2,1,0))#将RGB变换到BGR
'''

import caffe
import numpy as np
import matplotlib.pyplot as plt
import os
import PIL
from PIL import Image
import sys


# # googlenet网络结构描述文件
# deploy_file = 'config/googlenet/deploy.prototxt'
# # 训练好的模型
# model_file = 'models/googlenet/bvlc_googlenet_iter_190000.caffemodel'

#caffenet网络结构描述文件
deploy_file = 'config/caffenet/deploy.prototxt'
#训练好的模型
model_file = 'models/caffenet/caffenet_train_iter_5000.caffemodel'

# 均值文件(若没有均值文件,则用[104, 117, 123]代替)
mean_file='mean/image_mean.npy'
if os.path.exists(mean_file):
    image_mean=np.load(mean_file).mean(1).mean(1)
else:
    image_mean = [104, 117, 123]


# gpu或cpu运行模式
#caffe.set_device(0)
# caffe.set_mode_gpu()
caffe.set_mode_cpu()

# 这里直接使用caffe封装好分类器进行图像分类,可以查看源码
net = caffe.Classifier(deploy_file,  # 调用deploy文件
                       model_file,  # 调用模型文件
                       mean=image_mean,# 调用均值文件
                       channel_swap=(2, 1, 0),  # caffe中图片是BGR格式，而原始格式是RGB，所以要转化
                       raw_scale=255,  # python中将图片存储为[0, 1]，而caffe中将图片存储为[0, 255]，所以需要一个转换
                       image_dims=(227, 227))  # 输入模型的图片要是227*227的图片

# 分类标签文件
imagenet_labels_filename = 'dataset/label.txt'

# 载入分类标签文件
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

# 对目标路径中的图像，遍历并分类
for root, dirs, files in os.walk('test_image/'):
    for file in files:
        # 加载要分类的图片
        image_path = os.path.join(root, file)
        input_image = caffe.io.load_image(image_path)

        # 打印图片路径及名称

        print('******************************************************')
        print(image_path)

        # 预测图片类别,输入数据要求float32 [0,255]
        # output_prob=array([[1.3032453e-08, 1.9998259e-10, 2.7216041e-07, 6.3294447e-10, 9.9999982e-01]], dtype=float32))
        output= net.predict([input_image])#输入数据float32,[0,1],所以raw_scale=255
        print(output)
        output_prob=output[0]
        print(output_prob)

        # 批处理中第一个图像的输出概率向量
        # output_prob = output['prob'][0]

        pre_label = output_prob.argmax()

        print('predicted class:%d-->%s' % (pre_label, labels[pre_label]))

        # 显示图片
        img = Image.open(image_path)
        plt.imshow(img)
        plt.title(str(pre_label) + ':' + str(labels[pre_label]))
        plt.axis('off')
        plt.show()
        # 输出概率最大的前5个预测结果
        top_k = output_prob.argsort()[-5:][::-1]
        for node_id in top_k:
            # 获取分类名称
            human_string = labels[node_id]
            # 获取该分类的置信度
            score = output_prob[node_id]
            print('%s (score = %.5f)' % (human_string, score))
