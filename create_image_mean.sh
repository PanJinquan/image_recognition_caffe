#Ref:http://www.cnblogs.com/denny402/p/5102328.html
#一、二进制格式的均值文件*.binaryproto计算方法
# 计算均值的文件的工具:compute_image_mean.cpp，放在caffe根目录下的tools文件夹里面
#参数说明:
#      -第一个参数: 表示需要计算均值的数据，格式为lmdb的训练数据。
#      -第二个参数： 计算出来的结果保存文件*.binaryproto
/home/ubuntu/caffe/build/tools/compute_image_mean \
dataset/train_lmdb \
mean/image_mean.binaryproto


#二、python格式的均值文件npy计算方法
# 如果我们要使用python接口，或者我们要进行特征可视化，可能就要用到python格式的均值文件了。
# 首先，我们用lmdb格式的数据，计算出二进制格式的均值文件(*.binaryproto)，
# 然后，再将二进制格式的均值文件(*.binaryproto)转换成python格式(*.npy)的均值
# 工程:tools/convert_mean.py可以实现这个转换:
# 使用方法: python convert_mean.py path/to/mean.binaryproto /where/to/save.npy
python tools/convert_mean.py mean/image_mean.binaryproto mean/image_mean.npy