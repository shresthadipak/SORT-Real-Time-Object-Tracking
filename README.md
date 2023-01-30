# Real Time Object Tracking using SORT and openCV
Real-time object tracking refers to the process of locating and moving objects in a video stream using computer vision algorithms. This can be done using various techniques such as background subtraction, [Kalman filters](https://www.youtube.com/playlist?list=PLX2gX-ftPVXU3oUFNATxGXY90AULiqnWT), and deep learning-based object detection models. The goal of real-time object tracking is to accurately and efficiently locate and track objects in a video stream with minimal delay. This technology is used in a wide range of applications such as surveillance, self-driving cars, and augmented reality.

# YOLOv4 Object Detection Model
Object detection using YOLOv4 from scratch and have some basic concept over object detection model via the flow diagram.

![This is an image](/images/AO.png)

YOLOv4 is a convolutional neural network (CNN) based object detection model. It uses a single neural network to predict bounding boxes and class probabilities directly from full images in one pass. The architecture of YOLOv4 consists of several layers, including:

1. A backbone network, which is responsible for extracting feature maps from the input image. In YOLOv4, the backbone network is a variant of the CSPDarknet architecture, which is a combination of the Darknet and Cross Stage Partial (CSP) architectures.

2. A neck network, which is used to fuse feature maps from the backbone network and extract higher-level features. In YOLOv4, the neck network consists of several SPP (Spatial Pyramid Pooling) and PAN (Path Aggregation Network) blocks.

3. A head network, which is used to predict bounding boxes and class probabilities from the features extracted by the neck network. The head network in YOLOv4 consists of several YOLO (You Only Look Once) blocks, which are similar to the YOLOv3 blocks but with some modifications.

4. A auxiliary network, which is used to enhance the feature maps and improve the accuracy of the prediction, The auxiliary network in YOLOv4 consists of SPADE (Spatially Adaptive Normalization) blocks and PAN blocks.

Overall, YOLOv4 architecture is more efficient and accurate than YOLOv3.

[Download file yolov4 model](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) and save it into YOLOv4_model folder.

# License
The MIT License (MIT). Please see [License File](/LICENSE) for more information.