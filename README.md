# TAE-MVFI
本仓库是《基于Transformer和增强可变形可分离卷积的多视频插帧方法》的论文代码
## 依赖包
以下为运行代码所需要的依赖：
* python==3.7.6
* pytorch==1.5.1
* cudatoolkit==10.1
* torchvision==0.6.1
* cupy==7.5.0
* pillow==8.2.0
* einops==0.3.0
* opencv-python
* timm
* tqdm
## 涉及数据集
1. [Vimeo90K Triplet dataset](http://toflow.csail.mit.edu/)
2. [Vimeo90K Septuplet dataset](http://toflow.csail.mit.edu/)
3. [UCF101 dataset](https://www.google.com/url?q=https%3A%2F%2Fwww.dropbox.com%2Fs%2Fdbihqk5deobn0f7%2Fucf101_extracted.zip%3Fdl%3D0&sa=D&sntz=1&usg=AFQjCNE8CyLdENKhJf2eyFUWu6G2D1iJUQ)
4. [DAVIS dataset](https://www.google.com/url?q=https%3A%2F%2Fwww.dropbox.com%2Fs%2F9t6x7fi9ui0x6bt%2Fdavis-90.zip%3Fdl%3D0&sa=D&sntz=1&usg=AFQjCNG7jT-Up65GD33d1tUftjPYNdQxkg)
## 单帧插值
### 训练
```shell
python main.py --model TAE_MVFI_s --dataset vimeo90K_triplet --data_root <dataset_path> --batch_size 8 --num_workers 32
```
### 测试
1. Vimeo90K triplet
```shell
python test.py --model TAE_MVFI_s --dataset vimeo90K_triplet --data_root <dataset_path> --load_from checkpoints/TAE_MVFI_s/model_best.pth
```
2. UCF101
```shell
python test.py --model TAE_MVFI_s --dataset ucf101 --data_root <dataset_path> --load_from checkpoints/TAE_MVFI_s/model_best.pth
```
3. DAVIS
```shell
python test.py --model TAE_MVFI_s --dataset Davis --data_root <dataset_path> --load_from checkpoints/TAE_MVFI_s/model_best.pth
```
### 插帧
1. 单次插值：
```shell
python interpolate_demo.py --model TAE_MVFI_s --load_from checkpoints/TAE_MVFI_s/model_best.pth
```
2. 多次插值：
```shell
python interpolate_demo1.py --model TAE_MVFI_s --load_from checkpoints/TAE_MVFI_s/model_best.pth
```
## 多帧插值
### 训练
```shell
python main.py --model TAE_MVFI_m --dataset vimeo90K_septuplet --data_root <dataset_path> --batch_size 8 --num_workers 32
```
### 测试
1. Vimeo90K septuplet
```shell
python test.py --model TAE_MVFI_m --dataset vimeo90K_septuplet --data_root <dataset_path> --load_from checkpoints/TAE_MVFI_m/model_best.pth
```
2. UCF101
```shell
python test.py --model TAE_MVFI_m --dataset ucf101 --data_root <dataset_path> --load_from checkpoints/TAE_MVFI_m/model_best.pth
```
3. DAVIS
```shell
python test.py --model TAE_MVFI_m --dataset Davis --data_root <dataset_path> --load_from checkpoints/TAE_MVFI_m/model_best.pth
```
### 插帧
1. 指定插值中间帧的数量：
```shell
python interpolate_demo1.py --model TAE_MVFI_m --load_from checkpoints/TAE_MVFI_m/model_best.pth --inter_num 5
```
2. 指定插值中间帧对应的时间步：
```shell
python interpolate_demo2.py --model TAE_MVFI_m --load_from checkpoints/TAE_MVFI_m/model_best.pth --times 0.1,0.3,0.5,0.7,0.9
```
## 参考资料
本论文代码借鉴了以下论文开源代码，在此致谢：
* VFIT: Video Frame Interpolation Transformer, CVPR 2022 [Code](https://github.com/zhshi0816/Video-Frame-Interpolation-Transformer)
* VFIformer: Video Frame Interpolation with Transformer, CVPR 2022 [Code](https://github.com/dvlab-research/VFIformer)
* EDSC: Multiple Video Frame Interpolation via Enhanced Deformable Separable Convolution, PAMI 2021 [Code](https://github.com/Xianhang/EDSC-pytorch)
