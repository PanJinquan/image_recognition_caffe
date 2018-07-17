#caffe训练脚本

#/home/ubuntu/caffe/build/tools/caffe  train \
#--solver=config/googlenet/solver.prototxt \
#2>&1| tee models/log/caffe.log  #\
#--weights=models/googlenet/bvlc_googlenet.caffemodel

##(1) 2>&1| tee models/log/caffe.log
#  -命令tee ：将输出内容 重定向到日志文件中，同时在终端打印输出
#  -命令2>&1 是将标准出错重定向到标准输出，这里的标准输出已经重定向到了out.file文件，即将标准出错也输出到out.file文件中。
#  -命令&， 是让该命令在后台执行
##(2) 2>> models/log/caffe.log
/home/ubuntu/caffe/build/tools/caffe  train \
--solver=config/caffenet/solver.prototxt \
2>&1| tee models/log/caffe.log  #\
#--weights=models/caffenet/bvlc_googlenet.caffemodel