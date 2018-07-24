# -*- coding:utf-8 -*-

import h5py
import os
import cv2
import math
import numpy as np
import random
import re
import matplotlib.pyplot as plt

########################################################
def get_image_mean(image_dir,labels_path,mean_out,data_format):
    with open(labels_path, 'r') as f:
        lines = f.readlines()
    num = len(lines)
    channels, height, width=data_format
    # imgs = np.zeros([num,channels,height,width],dtype=np.float32)
    imgsMean=np.zeros([channels,height,width],dtype=np.float32)
    for i in range(num):
        line = lines[i]
        segments = re.split('\s+', line)[:-1]
        image_path=os.path.join(image_dir, segments[0])
        img = cv2.imread(image_path)
        if not os.path.exists(image_path):
            print('Err:图片不存在：',image_path)
            continue
        img = cv2.resize(img, (width, height))
        img = img.transpose(2, 0, 1)  #[h,w,c]->[c,h,w]
        # img = img[0, :, :].astype(np.float32)  #
        img = img.astype(np.float32)  #
        # imgs[i, :, :, :] = img.astype(np.float32)
        imgsMean=img/(num*1.0)+imgsMean    # 计算图像的均值
        if (i + 1) % 1000 == 0:
            print('Processed {} images!'.format(i + 1))

    # imgsMean2 = np.mean(imgs, axis=0)
    # imgsMean2 = np.mean(imgsMean2, axis=(1,2))
    # print('imgsMean2=',imgsMean2)
    # imgs = (imgs - imgsMean)/255.0
    # 保存均值文件
    means = np.mean(imgsMean, axis=(1,2))
    with open(mean_out, 'w') as f:
        f.write(str(means[0]) + '\n' + str(means[1]) + '\n' + str(means[2]))
    return imgsMean
########################################################
def get_imaes(image_dir,lines,label_num,data_format):
    num = len(lines)
    channels, height, width=data_format
    imgs = np.zeros([num, channels, height,width],dtype=np.float32)
    # imgs2 = np.zeros([num, 1, 100, 100])

    labels = np.zeros([num, label_num],dtype=np.float32)#10个label
    for i in range(num):
        line = lines[i]
        segments = re.split('\s+', line)[:-1]
        # print segments
        # 读取图像
        image_path=os.path.join(image_dir, segments[0])
        img = cv2.imread(image_path)
        img = cv2.resize(img, (width, height))#resize的第一个参数是W,第二个是H
        img = img.transpose(2, 0, 1)  #[h,w,c]->[c,h,w]
        # img = img[0, :, :].astype(np.float32)
        imgs[i, :, :, :] = img.astype(np.float32)

        # img2 = plt.imread(os.path.join(image_dir, segments[0]))
        # img2=img2[:, :, 0].astype(np.float32)#qu
        # img2 = img2.reshape((1,) + img2.shape)
        # imgs2[i, :, :, :] = img2.astype(np.float32)

        for j in range(label_num):
            labels[i, j] = float(segments[j + 1])

        if (i + 1) % 1000 == 0:
            print('Processed {} images!'.format(i + 1))

    return imgs,labels

########################################################
def gen_hdf5(labels_path,image_dir,batchSize,label_num,data_format,h5_path,means=0):
    setname, ext = h5_path.split('.')
    # 读取labels文件
    with open(labels_path, 'r') as f:
        lines = f.readlines()

    num = len(lines)
    # 打乱样本的数据
    # random.shuffle(lines)
    batchNum = int(math.ceil(1.0*num/batchSize))

    print("开始保存HDF5数据.....")
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
    for i in range(batchNum):
        start = i*batchSize
        end = min((i+1)*batchSize, num)
        batch_lines=lines[start:end]
        batch_imgs, batch_labels=get_imaes(image_dir,batch_lines,label_num,data_format)
        # filename = h5_path+'/train{0}.h5'.format(i)
        filename= setname+'{0}.h5'.format(i)
        print('save:%s'%(filename))
        with h5py.File(filename, 'w') as f:
            batch_imgs=batch_imgs-means   #减去均值(中心化)
            # batch_imgs=batch_imgs/255.0 #归一化到[0,255]
            # 参数comp_kwargs用于压缩数据
            # f.create_dataset('data', data = np.array((batch_imgs)).astype(np.float32), **comp_kwargs)
            # f.create_dataset('freq', data = np.array(batch_labels.astype(np.float32)), **comp_kwargs)
            f.create_dataset('data', data = np.array((batch_imgs)).astype(np.float32))
            f.create_dataset('labels', data = np.array(batch_labels.astype(np.float32)))

        # 保存hdf5文件列表
        h5_list='{}_h5.txt'.format(setname)
        with open(h5_list, 'a') as f:
            f.write(filename + '\n')


if __name__=='__main__':
    # 设置图片保存格式
    batchSize = 3000
    images_channel=3 #images_channel只能等于3
    resize_height=227
    resize_width=227
    data_format=[images_channel,resize_height,resize_width]#组成[C,H,W]
    # 当数据较多时,一般每3000个图片就存为一个*.h5文件
    label_num=12
    # 原始图像的路径
    image_dir = 'dataset/datasetImages_warp256'

    # 保存train HDF5数据
    root_path=os.getcwd()
    train_labels_path=os.path.join(root_path, 'dataset/multi_train.txt')#图像label文件的路径
    train_h5_out=os.path.join(root_path, 'dataset/HDF5/multi_train.h5') #保存h5文件的目录
    image_mean_out=os.path.join(root_path, 'dataset/mean/multi_train_mean.txt')#保存均值文件的路径

    # 首先计算训练样本均值
    print("正在计算图像均值.....")
    imgsMean=get_image_mean(image_dir=image_dir,
                            labels_path=train_labels_path,
                            mean_out=image_mean_out,
                            data_format=data_format)
    means_temp = np.mean(imgsMean, axis=(1,2))
    means = np.zeros([3, 1, 1])
    means[0, :, :] = means_temp[0]
    means[1, :, :] = means_temp[1]
    means[2, :, :] = means_temp[2]
    print("image mean:",means)
    # imgsMean=128

    gen_hdf5(labels_path=train_labels_path,
             image_dir=image_dir,
             batchSize=batchSize,
             label_num=label_num,
             data_format=data_format,
             h5_path=train_h5_out,
             means=means)


    # 保存val HDF5数据
    root_path=os.getcwd()
    val_labels_path=os.path.join(root_path, 'dataset/multi_val.txt')#图像label文件的路径
    val_h5_out=os.path.join(root_path, 'dataset/HDF5/multi_val.h5') #保存h5文件的目录
    gen_hdf5(labels_path=val_labels_path,
             image_dir=image_dir,
             batchSize=batchSize,
             label_num=label_num,
             data_format=data_format,
             h5_path=val_h5_out,
             means=means)