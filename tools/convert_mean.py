# coding: utf-8

#制作python格式的均值文件
import numpy as np
import sys,caffe

if len(sys.argv)==3:
    print "Usage: python convert_mean.py mean.binaryproto mean.npy"
    bin_file=sys.argv[1]
    out_file=sys.argv[2]
else:
    bin_file='image_mean.binaryproto'
    out_file='image_mean.npy'

blob = caffe.proto.caffe_pb2.BlobProto()

bin_mean = open(bin_file, 'rb').read()
blob.ParseFromString(bin_mean)
arr = np.array( caffe.io.blobproto_to_array(blob))
npy_mean = arr[0]
np.save(out_file, npy_mean)