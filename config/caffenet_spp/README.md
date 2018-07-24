#  caffenet_spp
------
## 增加spp层的方法
，spp层可以消除由于resize，crop等造成的精度损失。在CNN网络中，卷积层是不需要固定图像的大小（它的输出大小是跟输入图像的大小相关的），而全连接层是需要固定输入图像大小（全连接部分的参数的个数是需要固定的）。因此，在原来模型上，需要在第一个全连接前加上spp层：
```python
#layer {
#  name: "pool5"
#  type: "Pooling"
#  bottom: "conv5"
#  top: "pool5"
#  pooling_param {
#    pool: MAX
#    kernel_size: 3
#    stride: 2
#  }
#}

layer {
  name: "spatial_pyramid_pooling"
  type: "SPP"
  bottom: "conv5"
  top: "pool5"
  spp_param {
    pool: MAX
    pyramid_height: 2 # SPP的level的数量
  }
}
```
当输入数据的尺寸是固定时（如已经全部resize到227*277了），只需要Pooling换掉，只有一点注意，pyramid_height的设置如果过大（图片的size很小）会报的错误：Check failed: pad_h_ < kernel_h_，因为size太小了，到conv5时feature map太小了，而caffe这里限制pad_h_ < kernel_h_

当输入数据的尺寸是任意时，如果你的batch不是1的话，会报类似的错误：Check failed: height <= datum_height (80(第一个数据的height) vs. 64(后续数据的height))，这是因为load_batch是多线程同步的，caffe会默认用这个batch里的第一个数据的chanel height width作为输出的格式，所以如果不修改部分源码的话，只有先把trainval.prototxt和test.prototxt的batchsize都设为1