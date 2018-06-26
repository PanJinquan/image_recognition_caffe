::%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%格式转换的可执行文件%
%重新设置图片的大小%
%打乱图片%
%转换格式%
%图片路径%
%图片标签%
%lmdb文件的输出路径%
::%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%train%
D:\Caffe\caffe_cpu\bin\convert_imageset.exe ^
--resize_height=256 --resize_width=256 ^
--shuffle ^
--backend="lmdb" ^
dataset\train\ ^
dataset\train.txt ^
dataset\train_lmdb\
::%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%val%
D:\Caffe\caffe_cpu\bin\convert_imageset.exe ^
--resize_height=256 --resize_width=256 ^
--shuffle ^
--backend="lmdb" ^
dataset\val\ ^
dataset\val.txt ^
dataset\val_lmdb\
::%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pause
