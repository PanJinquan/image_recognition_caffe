# coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
import matplotlib.pyplot as plt
import os
import PIL
from PIL import Image
import caffe


#定义Caffe根目录
# caffe_root = 'E:/Caffe-windows/caffe-windows/'
#网络结构描述文件
deploy_file = 'config/caffenet/deploy.prototxt'
#训练好的模型
model_file = 'models/caffenet/caffenet_train_iter_5000.caffemodel'

#gpu模式
#caffe.set_device(0)
# caffe.set_mode_gpu()
caffe.set_mode_cpu()

#定义网络模型
net = caffe.Classifier(deploy_file, #调用deploy文件
                       model_file,  #调用模型文件
                       channel_swap=(2,1,0),  #caffe中图片是BGR格式，而原始格式是RGB，所以要转化
                       raw_scale=255,         #python中将图片存储为[0, 1]，而caffe中将图片存储为[0, 255]，所以需要一个转换
                       image_dims=(227, 227)) #输入模型的图片要是227*227的图片


#分类标签文件
imagenet_labels_filename = 'dataset/label.txt'
#载入分类标签文件
labels = np.loadtxt(imagenet_labels_filename, dtype=np.str, delimiter=' ')


#对目标路径中的图像，遍历并分类
for root,dirs,files in os.walk('test_image/'):
    for file in files:
        #加载要分类的图片
        image_file = os.path.join(root,file)
        input_image = caffe.io.load_image(image_file)

        #打印图片路径及名称
        image_path = os.path.join(root,file)
        print(image_path)

        #显示图片
        img=Image.open(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

        #预测图片类别
        prediction = net.predict([input_image])
        print('predicted class:',prediction[0].argmax())

        # 输出概率最大的前5个预测结果
        top_k = prediction[0].argsort()[::-1]
        for node_id in top_k:
            #获取分类名称
            human_string = labels[node_id]
            #获取该分类的置信度
            score = prediction[0][node_id]
            print("%s (score = %.5f)" %(human_string, score))


