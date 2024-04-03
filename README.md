# Computer-Vision-MPs
Deep Learning for Computer Vision-UIUC-CS course assignments

![DALLE_best](https://github.com/QiLong25/Computer-Vision-MPs/assets/143149589/8bd17801-3086-48a6-8155-01726b019bb0)


## MP1: Linear classifiers
[Rice Dataset](https://www.kaggle.com/datasets/mssmartypants/rice-type-classification)

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)

### Models
  * **Logistic Regression**.

  * **Perceptron**: binary / multi-class

  * **SVM**: binary / multi-class

  * **Softmax**: binary / multi-class

### Experiments
  * **Learning Rate**: learning rate decay strategy

  * **Training Epochs**: early stop

  * **Regularization Constant**.

  * **SGD Batch Size**.

  * **Stable Softmax**: avoid overflow

  * **Dataset Standardization**: center data and scale down by maximum

### Test Accuracy (Kaggle Fashion-MNIST)
  * Perceptron: 0.8182 (above baseline)

  * SVM: 0.8137 (above baseline)

  * Softmax: 0.8303 (above baseline)

## MP2: Multi-Layer Neural Networks

[Image Reconstruction](https://bmild.github.io/fourfeat/)

https://github.com/QiLong25/Computer-Vision-MPs/assets/143149589/a9d60fef-a049-45b6-b87f-515d37e43ca6

https://github.com/QiLong25/Computer-Vision-MPs/assets/143149589/57e2cbf1-4ce2-44c6-9969-0b9f7143f813

### Multi-Layer Neural Networks From Scratch

 *  **Linear Layer**: forward & backward.

 *  **ReLU**: forward & backward.

 *  **Sigmoid**: forward & backward.

 *  **MSE**: foward & backward.

 *  **MSE-Sigmoid**: forward & backward.

 *  **Forward**: four hidden layers of Linear + ReLU.

 *  **Backward**: MSE Loss, update parameters compute gradient.

 *  **Update**: optimizer (SGD, ADAM), momentum, update parameters.

### Image Reconstruction Helper Functions

 *  **Feature Mapping**: None, Basic, Gaussian. Fourier Mapping.

 *  **Hyperparameters**: num_layers, hidden_size, epochs, learning_rate, output_size.

 *  **Evaluation Metrics**: PSNR, MSE Loss.

### Experiment

####  **Pipeline**: 
 *  shuffle train_set, train, test.

####  **Optimizer & Mappings**:
 *  SGD:
![low-SGD-plot](https://github.com/QiLong25/Computer-Vision-MPs/assets/143149589/c531e541-3ee8-4038-ac84-23babb067aaa)

![low-SGD-img](https://github.com/QiLong25/Computer-Vision-MPs/assets/143149589/baec1093-707b-4304-8b67-eef775646c79)

 *  ADAM:
![low-Adam-plot](https://github.com/QiLong25/Computer-Vision-MPs/assets/143149589/750bc8da-80e2-4a27-904a-c7eeccec7eb0)

![low-Adam-img](https://github.com/QiLong25/Computer-Vision-MPs/assets/143149589/0d2a3539-3fe9-4909-bea0-434213f647be)

####  **High Resolution**:
 *  None:
 ![spiral-none](https://github.com/QiLong25/Computer-Vision-MPs/assets/143149589/81bb33d3-867b-4e49-aeef-5bab0ef4c5fd)

![spiral-none-img](https://github.com/QiLong25/Computer-Vision-MPs/assets/143149589/25fb0d04-bc34-45cf-a3c9-b2a03c696e56)

 *  Basic:
![spiral-basic-plot](https://github.com/QiLong25/Computer-Vision-MPs/assets/143149589/c3d708a0-9dbc-411d-a99a-252d49c39483)

![spiral-basic-img](https://github.com/QiLong25/Computer-Vision-MPs/assets/143149589/cefd6db0-35c0-4e08-9150-9749ccc17052)

 *  Fourier (Gaussian):
![spiral-fourier-plot](https://github.com/QiLong25/Computer-Vision-MPs/assets/143149589/89ded132-58e9-4bf0-a228-06bd43d8452b)

![spiral-fourier-img](https://github.com/QiLong25/Computer-Vision-MPs/assets/143149589/6e65c220-6a14-4f70-967e-3310b6b7166b)

#### **Number of Layers**.

#### **Fourier Feature Mapping Size**.

#### **Fourier Feature Mapping Gaussian Hyperparameters**.

#### **L1 Loss**.

#### **Regularization**.

#### **Normalization**.

### PyTorch Implementation.

*See Report.pdf for Further Details*

## MP3.1: Self-supervised & Transfer Learning

[CIFAR10](https://en.wikipedia.org/wiki/CIFAR-10)

[ImageNette](https://github.com/fastai/imagenette)

### Self-supervised Learning: Rotation Prediction

 *  Test Accuracy = 78.92%

### Finetune: Image classification, only last layers (layer4, fc)

 *  Pretrained Model: ResNet18 on **Rotation Prediction** task, Test Accuracy = 68.16%.

 *  Random-init Model: ResNet18 with weights initialized randomly, Test Accuracy = 44.45%.

### Fully Supervised Learning: Image classification

 *  Pretrained Model: ResNet18 on **Rotation Prediction** task, Test Accuracy = 84.35%.

 *  Random-init Model: ResNet18 with weights initialized randomly, Test Accuracy = 84.47%.

### Experiment on Accuracy-Sample

 *  Setup: large unlabelled dataset with a small amount of labeled data available.

 *  **RotNet**: Pretrain on **whole dataset** + Finetune (last layers) on **partial dataset**.

 *  **Supervised Model**: Fully Supervised Learning on **partial dataset**.

![EC1_plot](https://github.com/QiLong25/Computer-Vision-MPs/assets/143149589/bbd96558-5590-4352-acca-2842ebb68d25)

### More Advanced Model: Try ResNet50

 *  Pretrain Test Accuracy = 79.33%.

 *  Finetune Test Accuracy = 76.13%.

### Larger Dataset: Try ImageNette

 *  Pretrain Test Accuracy = 59.75%.

 *  Finetune Test Accuracy = 75.69%.

## MP3.2: Object Detection - YOLO

[PASCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007)

### Loss: Implement from scratch

 *  Setup: ResNet50, epochs=50.

 *  Final Train Loss: 1.751
 
 *  **Test MAP**: 0.50013

![yolo_image](https://github.com/QiLong25/Computer-Vision-MPs/assets/143149589/e0357f62-2b7a-4e70-9657-0b846f0298a1)

### Video Object Detection

![Video Source](https://www.youtube.com/watch?v=xZGahvrep3o)

https://github.com/QiLong25/Computer-Vision-MPs/assets/143149589/95a042ae-4c09-46df-b98e-72003fb7676d

### Better Pretrained Model

 *  Setup: ResNet101, epochs=50.

 *  Final Train Loss: 1.544

 *  **Test MAP**: 0.52478

![yolo_image_2](https://github.com/QiLong25/Computer-Vision-MPs/assets/143149589/fa757032-6d35-4ede-9fd5-19f528abba47)

*See Report.pdf for Further Details*































