#!/usr/bin/env sh
# Create the face_48 lmdb inputs
# N.B. set the path to the face_48 train + val data dirs
######################################################################

EXAMPLE=dataset  #%lmdb文件的输出路径
DATA=/home/ubuntu/caffeProject/image_recognition_caffe/dataset     #train.txt和val.txt 文件的位置
TOOLS=/home/ubuntu/caffe/build/tools                               #caffe格式转换的可执行文件，即convert_imageset的位置

TRAIN_DATA_ROOT=/home/ubuntu/caffeProject/image_recognition_caffe/dataset/train/ #train原始数据图片集的路径
VAL_DATA_ROOT=/home/ubuntu/caffeProject/image_recognition_caffe/dataset/val/     #val原始数据图片集的路径
######################################################################

# Set RESIZE=true to resize the images to 60 x 60. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=227
  RESIZE_WIDTH=227
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_face_48.sh to the path" \
       "where the face_48 training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_face_48.sh to the path" \
       "where the face_48 validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $EXAMPLE/train_lmdb
echo "××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××"

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/val.txt \
    $EXAMPLE/val_lmdb

echo "Done."
Status API Training Shop Blog About

