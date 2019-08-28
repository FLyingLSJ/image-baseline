# Basic models in industrial inspection

## ImageClassification
The classification models mainly based on CBAM-keras, the others are models that i explore and try:

* Frame include tensorflow and pytorch
* Specific details you can view README.md or TRAINING.md under each floder
* the structure of CBAM : [Convolutional Block Attention Module"](https://arxiv.org/pdf/1807.06521)
* The main project file I use = `./ImageClassClassification/CBAM-tensorflow-slim`


### CBAM_block and SE_block Supportive Models
- Inception V4 + CBAM / + SE
- Inception-ResNet-v2 + CBAM / + SE
- ResNet V1 50 + CBAM / + SE
- ResNet V1 101 + CBAM / + SE
- ResNet V1 152 + CBAM / + SE
- ResNet V1 200 + CBAM / + SE
- ResNet V2 50 + CBAM / + SE
- ResNet V2 101 + CBAM / + SE
- ResNet V2 152 + CBAM / + SE
- ResNet V2 200 + CBAM / + SE

### Requirements
- Python 3.x
- TensorFlow 1.x
- TF-slim
- torch 1.x
- Keras (IMDB dataset)
- tqdm
- scikit-image
- numpy
- torch>=0.4.0
- torchvision
- pillow
- matplotlib      
- [wing](https://wingware.com/)


## TargetDetection
The application scenario of the model is vehicle detection.There are SSD and yolov3
In the project, the model i used is `MyYOLO`. Other floders are versions on `keras` and `pytorch`.

* Result: When the confidence is 0.8, the accuracy rate is above 0.95.
* SSD is an unified framework for object detection with a single network. It has been originally introduced in this research [article](http://arxiv.org/abs/1512.02325).
* [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Original Implementation]](https://github.com/pjreddie/darknet)
* The main project file I use = `./TargetDetection/MyYOLO`


## SemantemeDivision
Semantic segmentation in cable quality inspection and autonomous driving applications
Irregular boundary lines by semantic segmentation to achieve quality inspection
the main network in the project is Bisenet+resnet50
* [network structure](https://pic3.zhimg.com/v2-d9b76a478043f7d68a1d452078654aee_r.jpg), [aritcle of Bisenet](https://arxiv.org/pdf/1808.00897.pdf)
* The main project file I use = `./SemantemeDivision/Segmentation`
* Supported models:
	* Fontends
		* Inceptions_v4
		* Mobilenet_v2
		* Resnet_v1
		* Resnet_v2
		* Se_resnext
	* Builders: 
		* Bisenet
		* Deeplab_v3
		* Refinenet