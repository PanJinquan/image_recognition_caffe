# image_recognition_caffe

# 一.制作lmdb数据
- create_lmdb_v2.sh


# 二.生成均值文件
- create_image_mean.sh

# 三.修改配置文件
- config/googlenet
- config/caffenet

# 四.训练模型
- train_linux.sh

# 五.测试模型
- image_recognition_test.py