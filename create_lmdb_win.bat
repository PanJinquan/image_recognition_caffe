::%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%��ʽת���Ŀ�ִ���ļ�%
%��������ͼƬ�Ĵ�С%
%����ͼƬ%
%ת����ʽ%
%ͼƬ·��%
%ͼƬ��ǩ%
%lmdb�ļ������·��%
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
