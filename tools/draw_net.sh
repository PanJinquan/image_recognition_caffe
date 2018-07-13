# draw_net.py
# 第一个参数：网络模型的prototxt文件
# 第二个参数：保存的图片路径及名字
# 第三个参数：--rankdir=x , x 有四种选项，分别是LR, RL, TB, BT 。用来表示网络的方向，分别是从左到右，从右到左，从上到小，从下到上。默认为LR。
# e.g.: python  caffe/python/draw_net.py  path/to/*.prototxt  where/to/save.png
# 依赖库:
# pip install pydot
# sudo apt-get install graphviz  
# 
python /home/ubuntu/caffe/python/draw_net.py config/caffenet/train_val.prototxt googlenet.png --rankdir=BT

# 可以使用这个网址的脚本绘制网络结构:
# Use Shift+Enter to update the visualization.
echo "http://ethereon.github.io/netscope/#/editor"