# State of the art detectors

This document provides a quick review of the most influential papers on object detection (i.e., classification and
bounding box estimation). It has a secondary focus on the backbone networks used, and whether or not the weights of the
detection models were transfered from other models pre-trained on other tasks and datasets. Also, low-latency /
low-hardware-requirements models are investigated more in-depth.

The papers mentioned here fall in either of the three following categories (although some papers fall into two
categories, though, as you will see below): *Top Performers*, *Seminal Works* and *High Speed*.

## 1. Top Performers

Papers that, at the time of this writing, yield the best results in terms of detection accuracy on the detection task on
the Microsoft COCO Dataset, according to the competition's
[official leaderboard]((http://cocodataset.org/#detection-leaderboard))

##### 1.a. Megvii (Face++)
- 1st place in the COCO leaderboard
- ResNeXt backbone (pretrained on ImageNet classification)

##### 1.b. PANet
- 2nd place in the COCO leaderboard, under the name "UCenter":
- [Liu, Qi, Qin, Shi, Jia (Hong Kong) - Path Aggregation Network for Instance Segmentation (Mar/Sep 2018)](https://arxiv.org/abs/1803.01534)
- based on Mask-RCNN (pre-trained on COCO)
- ResNet / ResNeXt backbones (pretrained on ImageNet classification)

##### 1.c. Deformable ConvNets
- 3rd place in the COCO leaderboard, under the name "MSRA"
- [Dai, Qi, Xiong, Li, Zhang, Hu, Wei (Microsoft) - Deformable Convolutional Networks (Mar/Jun 2017)](https://arxiv.org/abs/1703.06211)
- ResNet-101, Inception-ResNet backbones, pretrained on ImageNet classification
- implemented using mxnet
- follow-up paper with better results:
    - [Zhu, Hu, Lin, Dai (Microsoft / USTC) - Deformable ConvNets v2: More Deformable, Better Results (Nov 2018)](https://arxiv.org/abs/1811.11168)
    - also implemented using mxnet

## 2. Seminal works

Reference papers which introduced techniques used by many other works (including the top performers above) thus being
given historical importance.

##### 2.a. Mask R-CNN
- 4th place in the COCO leaderboard, under the name "FAIR Mask R-CNN"
- [He, Gkioxari, Dollár, Girshick (Facebook FAIR) - Mask R-CNN (Mar 2017 / Jan 2018)](https://arxiv.org/abs/1703.06870)
- can also be considered to fall in the *Top Performers* category above
- ResNet-50 / ResNext-101 backbone (pre-trained on ImageNet classification)
- based on Faster R-CNN:
    - 22nd place in the COCO leaderboard, under the name "MSRA"
    - [Ren, He, Girshick, Sun (Microsoft) - Faster R-CNN (Jun 2015 / Jan 2016)](https://arxiv.org/abs/1506.01497)
    - based on Fast R-CNN: [Girshick (Microsoft) - Fast R-CNN (Apr / Sep 2015)](https://arxiv.org/abs/1504.08083)
        - based on R-CNN: [Girshick, Donahue, Darrell, Malik (Berkeley) - Rich feature hierarchies for accurate object detection and semantic segmentation (Nov 2013 / Oct 2014)](https://arxiv.org/abs/1311.2524)
        
##### 2.b. FPN
- 24th place in the COCO Leaderboard (single model, i.e., no model ensambles)
- [Lin, Dollár, Girshick, He, Hariharam Belongie (Facebook FAIR) - Feature Pyramid Networks for Object Detection (Dec 2016 / Apr 2017)](https://arxiv.org/abs/1612.03144)
- based on SSD (see below)
- explained [here](https://medium.com/@jonathan_hui/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c)
- introduces a (very good) feature extractor (detection output layers are just the Faster R-CNN Predictor Head)
- very important: allows for sharing the same detector head across multiple scales, which means you take better
advantage of the training data (pos/neg is very imbalanced, anyway; there are very few positive boxes, e.g., on a small
scale, so this makes better use of this scarce resource)
    - another recent paper that allows detector head sharing over different output scales (or, conversely, input
    resolutions) is Google's PPN:
        - [Jin, Rathod, Zhu (Google) - Pooling Pyramid Network for Object Detection (Jul 2018)]((https://arxiv.org/abs/1807.03284))
- ResNet-50 / ResNet-101 backbone, pretrained on Imagenet classification

##### 2.c. YOLO
- v1: [Redmon, Divvala, Girschick, Farhadi (Washington, Facebook FAIR) - You Only Look Once: Unified, Real-Time Object Detection (Jun 2015 / May 2016)](https://arxiv.org/abs/1506.02640)
- v2: [Redmon, Farhadi (Washington) - YOLO9000: Better, Faster, Stronger (Dec 2016)](https://arxiv.org/abs/1612.08242)
    - 42nd place in clasamentul COCO, sub numele "Darknet"
- v3: [Redmon, Farhadi (Washington) - YOLOv3: An Incremental Improvement (Apr 2018)](https://arxiv.org/abs/1804.02767)
- all of them: original backbone architecture, pretrained on ImageNet classification

##### 2.d. R-FCN
- 33d place in the COCO leaderboard, under the name "ION"
- [Bell, Zitnick, Bala, Girshick (Cornell / Facebook) - Inside-Outside Net:
Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks
(Dec 2015)](https://arxiv.org/abs/1512.04143)
- explained [here](https://medium.com/@jonathan_hui/understanding-region-based-fully-convolutional-networks-r-fcn-for-object-detection-828316f07c99)
- good speed (they say it's about 20 times faster than RCNN), so it can be considered to also fall into the *High Speed*
category, below
- [follow-up](https://arxiv.org/abs/1712.01802)

## 3. High Speed

Papers which focus on obtaining real-time detection on mobile devices or inside browsers (limited resources: memory,
cpu, gpu, inference time).

### 3.a. State of the art
- [Huang et al. (Google) - Speed / accuracy trade-offs for modern convolutional
object detectors (Nov 2016 / Apr 2017)](https://arxiv.org/pdf/1611.10012.pdf)
- excellent resource to help decide what kind of model to use, given the hardware and latency constraints of your
application
- the academic result of the effort to build
[TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
- since it's been published, the TF OD API published, in its
[Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
results for some new networks, notably the following two, which I feel are good candidates to replace the originally
published "sweet spot" networks (i.e., the ones for which better accuracy incurs much higher computational cost than
the one needed to get them). Both of them use FPN for feature extraction and Focal Loss for training (both of them are
Facebook FAIR's brainchildren). Here's the the [paper about focal loss](https://arxiv.org/abs/1708.02002) and an
[explanation](https://medium.com/@14prakash/the-intuition-behind-retinanet-eb636755607d).
    - ssd_mobilnet_v1_fpn_coco (the speed of which can be easily improved by switching the backend from mobilenet v1 to
      v2, and from ssd to ssdlite)
    - ssd_resnet_50_fpn_coco (the speed of which can be - marginally - improved by switching to ssdlite)
    - both of them use backbones (mobilenetv1 and resnet-50) pretrained on ImageNet classification

### 3.b. SSD
- [Liu, Erhan, Szegedy, Reed, Fu, Berg (Chapel Hill, Zoox, Google, Ann-Arbor) - SSD: Single Shot MultiBox Detector (Dec 2015 / Dec 2016)](https://arxiv.org/abs/1512.02325)
- widely cited and used, so it can also be consider to fall under the category *Seminal Works*, above
- VGG16 backbone trained on ImageNet classification + localization
- can be easily attached to any backbone (backbones reportedly used in literature and elsewhere in opensource code
include: ResNet, MobileNet v1/v2, Inception v2/v3, Inception ResNet v2, SqueezeNet, ShuffleNet v1/v2):
    - MobileNet v1: [Howard et al. (Google) - MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (Apr 2017)](https://arxiv.org/abs/1704.04861)
    - shufflenet + ssd: [1](https://github.com/linchaozhang/shufflenet-ssd),
    [2](https://github.com/hemuwei/shufflenet_ssd), and [this](https://stackoverflow.com/q/52525372), which uses the
    code [here](https://github.com/TropComplique/shufflenet-v2-tensorflow) with TF OD API (for SSD)
- detectors based on SSD:
    - FPN (listed at 2.b. above)
    - SSDLite:
        - published with MobileNetV2: [Sandler, Howard et al. (Google) - MobileNetV2: Inverted Residuals and Linear Bottlenecks (Jan / Apr 2018)](https://arxiv.org/abs/1801.04381)
        - essentially replaces normal convolutions inside the detection / classification output layers with separable ones (depthwise + pointwise)
    - ShuffleDet:
        - [Azimi (German Aerospace) - ShuffleDet: Real-Time Vehicle DetectionNetwork in On-board Embedded UAV Imagery](https://arxiv.org/abs/1811.06318v1)
        - essentially, replaces normal convolutions inside the detection / classification output layers with shufflenet blocks
        - shufflenet backbone, pretrained on ImageNet classification
    - based on Fire modules:
        - Fire modules were introduced by SqueezeNet (classifier): [Iandola, Han, Moskewicz, Ashraf, Dally, Keutzer
            (Berkeley / Stanford) - SqueezeNet: AlexNet-level accuracy with 50x
            fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)
        - [Tiny SSD](https://arxiv.org/abs/1802.06488)
        - [Fire SSD (intel)](https://arxiv.org/abs/1806.05363)
    - Pelee:
        - [Wang, Li, Ling (Ontario) - Pelee: A Real-Time Object Detection System on Mobile Devices (NIPS, Apr 2018 / Jan 2019)](https://arxiv.org/abs/1804.06882)
        - Pretrained Atrous VGG
        - based on [DenseNet](https://arxiv.org/abs/1608.06993) (TODO: read)

##### 3.c. ShuffleNet:
- v1: [Zhang, Zhou, Lin, Sun (Megvii Face++) - ShuffleNet: An Extremely Efficient
Convolutional Neural Network for Mobile Devices (Jul / Dec 2017)](https://arxiv.org/abs/1707.01083)
    - original paper reports results using Faster R-CNN detector head
- v2: [Ma, Zhang, Zheng, Sun (Megvii Face++ / Tsinghua) - ShuffleNet V2:
Practical Guidelines for Efficient CNN Architecture Design (Jul 2018)](https://arxiv.org/abs/1807.11164)
    - pretrained on ImageNet classification (5k classes)
    - uses [Light-Head RCNN](https://arxiv.org/abs/1711.07264) (by Megvii/Face++) as detector head

##### 3.d. YOLO-LITE:
- [Huang, Pedoeem, Chen (Georgia Tech) - YOLO-LITE: A Real-Time Object DetectionAlgorithm Optimized for Non-GPU Computer (Nov 2018, IEEE Big Data)](https://arxiv.org/abs/1811.05588)
- no pretraining, but in *Future Work* section, they mention they want to pretrain (they have rather poor results, and
they think that using pretrained models will help improve them)

##### 3.e. SqueezeDet:
- [Wu, Wan, Iandola, Jin, Keutzer (Berkeley, DeepScale) - SqueezeDet: Unified, Small, Low Power Fully Convolutional Neural Networks for Real-Time Object Detection for Autonomous Driving (CVPR Workshops Conference, Dec 2016 / Nov 2017)](https://arxiv.org/abs/1612.01051)
- pre-trained on ImageNet classification

##### 3.f. PVANet:
- [Kim, Hong, Roh, Cheon, Park (Intel) - PVANET: Deep but Lightweight Neural Networks for Real-time Object Detection (Aug / Sep 2016 - EMDNN 2016 NIPS Workshop)](https://arxiv.org/abs/1608.08021)
- pre-trained on ImageNet classification

##### 3.g. TinyYolo:
- darknet, part of YoloV1, V2


## Other stuff:
- good, (very) recent paper about the effects of pretraining: [He, Girshick, Dollár (Facebook FAIR) - Rethinking ImageNet Pre-training (Nov 2018)](https://arxiv.org/abs/1811.08883)
    - says that pre-training does not influence model accuracy, just training time
    
