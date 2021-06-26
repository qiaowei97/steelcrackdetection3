代码地址：https://github.com/qiaowei97/steelcrackdetection3
模型地址：链接：https://pan.baidu.com/s/13YvQgQLJypf-WQcFiDZ9Eg  提取码：1234
系统：Windows/Ubuntu
硬件要求：NVIDIA显卡，显存 >= 4G
环境：CUDA 9.0，Pytorch 1.5.0，Numpy，prefetch_generator
需安装包：pytorch, numpy, os, pillow, torchvision, cv2
训练操作方式：将训练图像放入文件夹'_raw_data/train/'，运行代码'./utils/crop.py'对图像进行resize和裁剪；再运行'./data_aug.py'，对数据集进行扩充；运行'train.py'进行训练
测试操作方式：将训练图像放入文件夹'_raw_data/val/'，将模型文件'CP_full.pth'文件放在checkpoints文件夹内，运行'predict_full.py'代码即可
