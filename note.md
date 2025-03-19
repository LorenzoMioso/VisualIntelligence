# Key points discovered

- The class average color is sufficient for the model to classify the images with high accuracy, we need to focus on the structural features
- The dataset is balanced and the images are size 768x768 pixels, keeping the oringinal size because of the small cell structures
- With grayscale images, the model is able to classify the images with high accuracy
- CNN is good at learning even with a small classifier layer
- Scatnet needs more complex classifier layer
- CNN reaces 99% accuracy
- Scatnet reaches 92% accuracy
- CNN is faster than Scatnet
- using a low lr=1e-4 for cnn does not make filters converge, using lr=1e-3 makes the filters converge

# CNN architecture

==========================================================================================
Layer (type:depth-idx) Output Shape Param #
==========================================================================================
CNNImageClassifier [1, 2] --
├─Sequential: 1-1 [1, 24, 4, 4] --
│ └─Conv2d: 2-1 [1, 16, 382, 382] 1,952
│ └─BatchNorm2d: 2-2 [1, 16, 382, 382] 32
│ └─ReLU: 2-3 [1, 16, 382, 382] --
│ └─MaxPool2d: 2-4 [1, 16, 191, 191] --
│ └─Conv2d: 2-5 [1, 16, 191, 191] 2,320
│ └─BatchNorm2d: 2-6 [1, 16, 191, 191] 32
│ └─ReLU: 2-7 [1, 16, 191, 191] --
│ └─MaxPool2d: 2-8 [1, 16, 95, 95] --
│ └─Conv2d: 2-9 [1, 24, 95, 95] 3,480
│ └─BatchNorm2d: 2-10 [1, 24, 95, 95] 48
│ └─ReLU: 2-11 [1, 24, 95, 95] --
│ └─AdaptiveAvgPool2d: 2-12 [1, 24, 4, 4] --
├─FeatureClassifier: 1-2 [1, 2] --
│ └─Linear: 2-13 [1, 16] 6,160
│ └─BatchNorm1d: 2-14 [1, 16] 32
│ └─ReLU: 2-15 [1, 16] --
│ └─Dropout: 2-16 [1, 16] --
│ └─Linear: 2-17 [1, 2] 34
==========================================================================================
Total params: 14,090
Trainable params: 14,090
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 400.89
==========================================================================================
Input size (MB): 2.36
Forward/backward pass size (MB): 50.16
Params size (MB): 0.06
Estimated Total Size (MB): 52.58
==========================================================================================

## CNN Stats:

Fold 0 - Accuracy: 0.9760, F1 Score: 0.9764
Fold 1 - Accuracy: 0.9760, F1 Score: 0.9764
Fold 2 - Accuracy: 0.9920, F1 Score: 0.9920
Fold 3 - Accuracy: 0.9930, F1 Score: 0.9930
Fold 4 - Accuracy: 0.9820, F1 Score: 0.9823
Fold 5 - Accuracy: 0.9930, F1 Score: 0.9930
Fold 6 - Accuracy: 0.9950, F1 Score: 0.9950
Fold 7 - Accuracy: 0.9910, F1 Score: 0.9910
Fold 8 - Accuracy: 0.9960, F1 Score: 0.9960
Fold 9 - Accuracy: 0.9930, F1 Score: 0.9930

Mean Accuracy: 0.9887
Mean F1 Score: 0.9888

## ScatNet Stats:

Fold 0 - Accuracy: 0.9110, F1 Score: 0.9079
Fold 1 - Accuracy: 0.9260, F1 Score: 0.9270
Fold 2 - Accuracy: 0.8870, F1 Score: 0.8781
Fold 3 - Accuracy: 0.8890, F1 Score: 0.8797
Fold 4 - Accuracy: 0.8820, F1 Score: 0.8728
Fold 5 - Accuracy: 0.8110, F1 Score: 0.7736
Fold 6 - Accuracy: 0.8500, F1 Score: 0.8275
Fold 7 - Accuracy: 0.8780, F1 Score: 0.8671
Fold 8 - Accuracy: 0.8890, F1 Score: 0.8817
Fold 9 - Accuracy: 0.8650, F1 Score: 0.8524

Mean Accuracy: 0.8788
Mean F1 Score: 0.8668
