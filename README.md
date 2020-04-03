# mask-rcnn论文复现
## 模型
InceptionV2拼成的GoogLeNet，去掉全连接，取16倍降采样的卷积层。因为电脑显存不够，实在装不下论文上的resnet50+金字塔模型...
## 训练
训练用的是反复横跳的训练方式</br>
1. 用rpn_conv+rpn_net训练一个rpn网络。</br>
2. 用上面训练好的rpn锁住参数，训练一个rcnn_conv+rcnn_net的rcnn网络。</br>
3. 用rpn_conv加载rcnn_conv的参数，重新训练rpn_net，此时把rpn_conv的参数锁住，只训练rpn_net。</br>
4. 用锁住参数的rpn_conv+rpn_net重新训练rcnn_net，此时，rcnn_conv已经没用了，本来就是个工具人，起中转作用的。
## 效果
原图

目标检测+实例分割

最终效果

