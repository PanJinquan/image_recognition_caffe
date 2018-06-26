# coding: utf-8
import os
#定义数据集的根目录
# caffe_root = '/home/ubuntu/caffeProject/recognition/dataset/'

caffe_root=os.getcwd()+"/dataset/"
#制作训练标签数据
i = 0 #标签
with open(caffe_root + 'train.txt','w') as train_txt:
    for root,dirs,files in os.walk(caffe_root+'train/'): #遍历文件夹
        for dir in dirs:
            for root,dirs,files in os.walk(caffe_root+'train/'+str(dir)): #遍历每一个文件夹中的文件
                for file in files:
                    image_file = str(dir) + '/' + str(file)
                    label = image_file + ' ' + str(i) + '\n'       #文件路径+空格+标签编号+换行
                    train_txt.writelines(label)                   #写入标签文件中
                i+=1 #编号加1


#制作测试标签数据
i=0 #标签
with open(caffe_root + 'val.txt','w') as test_txt:
    for root,dirs,files in os.walk(caffe_root+'val/'): #遍历文件夹
        for dir in dirs:
            for root,dirs,files in os.walk(caffe_root+'val/'+str(dir)): #遍历每一个文件夹中的文件
                for file in files:
                    image_file = str(dir) + '/' + str(file)
                    label = image_file + ' ' + str(i) + '\n'       #文件路径+空格+标签编号+换行
                    test_txt.writelines(label)                   #写入标签文件中
                i+=1#编号加1

print("成功生成文件列表")



