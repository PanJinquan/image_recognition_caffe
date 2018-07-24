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
# # load the mean ImageNet image (as distributed with Caffe) for subtraction
# mu = np.load(mean_file)
# mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
# print 'mean-subtracted values:', zip('BGR', mu)


# gpu或cpu运行模式
#caffe.set_device(0)
# caffe.set_mode_gpu()
caffe.set_mode_cpu()

# 定义网络模型
net = caffe.Net(deploy_file,      # defines the structure of the model
                model_file,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

'''
 python读取的图片文件格式为H×W×K，需转化为K×H×W
 如原始图片(227,227,3),则需要set_transpose改变维度的顺序为(3,227,227)
'''
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension将图像通道移动到最外层
transformer.set_mean('data', image_mean)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR


# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(1,         # batch size,批量处理个个数,起对应输出output['prob'][i]的个数
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227


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
        #################################################################
        transformed_image = transformer.preprocess('data', input_image)

        # copy the image data into the memory allocated for the net
        net.blobs['data'].data[...] = transformed_image

        # perform classification
        output = net.forward()
        # output=net.forward_all(data=transformed_image)
        (out_name,out_value),=output.items()
        # items()获得字典的(key:lalue),其中out_name='prob'
        print('output=',out_name,out_value)

        # the output probability vector for the first image in the batch
        # 批处理中第一个图像的输出概率向量
        output_prob = output[out_name][0]# out_value
        print('output_prob=',output_prob)

        pre_label=output_prob.argmax()

        print('predicted class:%d-->%s' % (pre_label, labels[pre_label]))
        #
        # # 显示图片
        # img = Image.open(image_path)
        # plt.imshow(img)
        # plt.title(str(pre_label) + ':' + str(labels[pre_label]))
        # plt.axis('off')
        # plt.show()
        # # 输出概率最大的前5个预测结果
        # top_k = output_prob.argsort()[-5:][::-1]
        # for node_id in top_k:
        #     # 获取分类名称
        #     human_string = labels[node_id]
        #     # 获取该分类的置信度
        #     score = output_prob[node_id]
        #     print('%s (score = %.5f)' % (human_string, score))
