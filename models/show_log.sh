#训练loss和accuracy可视化,从log中提取准确率和误差等信息
#1.获得log文件:2>&1| tee models/log/caffe.log 或者2>> models/log/caffe.log 
#2.把caffe/tools/extra目录下的plot_training_log.py.example复制一份并重命名为plot_training_log.py(可以复制到工程目录，)
#3.python2.7执行plot_training_log.py:
#Usage:
#    ./plot_training_log.py chart_type[0-7] /where/to/save.png /path/to/first.log ...
#Notes:
#    1. Supporting multiple logs.
#    2. Log file name must end with the lower-cased ".log".
#Supported chart types:
#    0: Test accuracy  vs. Iters
#    1: Test accuracy  vs. Seconds
#    2: Test loss  vs. Iters
#    3: Test loss  vs. Seconds
#    4: Train learning rate  vs. Iters
#    5: Train learning rate  vs. Seconds
#    6: Train loss  vs. Iters
#    7: Train loss  vs. Seconds
#
#   -第一个参数:0-7,训练或者测试的accuracy,loss等数据
#   -第二个参数:图片存放的位置
#   -第三个参数:log文件
#   -      eg:python /home/ubuntu/caffe/tools/extra/plot_training_log.py 0 test.png
#
python /home/ubuntu/caffe/tools/extra/plot_training_log.py 0  log/Test-acc-Iters.png log/caffe.log
python /home/ubuntu/caffe/tools/extra/plot_training_log.py 1  log/Test-acc-Seconds.png log/caffe.log
python /home/ubuntu/caffe/tools/extra/plot_training_log.py 2  log/Test-loss-Iters.png log/caffe.log
python /home/ubuntu/caffe/tools/extra/plot_training_log.py 3  log/Test-loss-Seconds.png log/caffe.log
python /home/ubuntu/caffe/tools/extra/plot_training_log.py 4  log/Train-learning-rate-Iters.png log/caffe.log
python /home/ubuntu/caffe/tools/extra/plot_training_log.py 5  log/Train-learning-rate-Seconds.png log/caffe.log
python /home/ubuntu/caffe/tools/extra/plot_training_log.py 6  log/Train-loss-Iters.png log/caffe.log
python /home/ubuntu/caffe/tools/extra/plot_training_log.py 7  log/Train-loss-Seconds.png log/caffe.log