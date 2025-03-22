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

CNNImageClassifier(
(features): Sequential(
(0): Conv2d(1, 16, kernel_size=(11, 11), stride=(2, 2), padding=(3, 3))
(1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(2): ReLU()
(3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(4): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(5): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(6): ReLU()
(7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(8): Conv2d(16, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(9): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(10): ReLU()
(11): AdaptiveAvgPool2d(output_size=(4, 4))
)
(classifier): FeatureClassifier(
(fc1): Linear(in_features=384, out_features=16, bias=True)
(bn): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu): ReLU()
(do): Dropout(p=0.5, inplace=False)
(fc2): Linear(in_features=16, out_features=2, bias=True)
)
)

# ScatNet architecture

ScatNetImageClassifier(
(scattering): Scattering2D()
(global_pool): AdaptiveAvgPool2d(output_size=(4, 4))
(classifier): FeatureClassifier(
(fc1): Linear(in_features=3472, out_features=16, bias=True)
(bn): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu): ReLU()
(do): Dropout(p=0.5, inplace=False)
(fc2): Linear(in_features=16, out_features=2, bias=True)
)
)

## CNN Stats:

Evaluating model performance across all folds...
Loading model from checkpoint 9
Fold 0 - Accuracy: 0.9800, F1 Score: 0.9801
Fold 1 - Accuracy: 0.9770, F1 Score: 0.9774
Fold 2 - Accuracy: 0.9940, F1 Score: 0.9940
Fold 3 - Accuracy: 0.9970, F1 Score: 0.9970
Fold 4 - Accuracy: 0.9960, F1 Score: 0.9960
Fold 5 - Accuracy: 0.9960, F1 Score: 0.9960
Fold 6 - Accuracy: 0.9960, F1 Score: 0.9960
Fold 7 - Accuracy: 0.9930, F1 Score: 0.9930
Fold 8 - Accuracy: 0.9980, F1 Score: 0.9980
Fold 9 - Accuracy: 0.9990, F1 Score: 0.9990

Mean Accuracy: 0.9926 ± 0.0072
Mean F1 Score: 0.9927 ± 0.0072
Model evaluation completed.

## ScatNet Stats:

Evaluating model performance across all folds...
Loading model from checkpoint 9
Fold 0 - Accuracy: 0.9210, F1 Score: 0.9209
Fold 1 - Accuracy: 0.8910, F1 Score: 0.8868
Fold 2 - Accuracy: 0.9290, F1 Score: 0.9279
Fold 3 - Accuracy: 0.9250, F1 Score: 0.9208
Fold 4 - Accuracy: 0.9380, F1 Score: 0.9371
Fold 5 - Accuracy: 0.9440, F1 Score: 0.9443
Fold 6 - Accuracy: 0.9510, F1 Score: 0.9506
Fold 7 - Accuracy: 0.9380, F1 Score: 0.9358
Fold 8 - Accuracy: 0.9230, F1 Score: 0.9197
Fold 9 - Accuracy: 0.9390, F1 Score: 0.9388

Mean Accuracy: 0.9299 ± 0.0159
Mean F1 Score: 0.9283 ± 0.0170
Model evaluation completed.
