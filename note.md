# Key points discovered

- The model is learning the color of the class -> turnimg the image to gray scale
- CNN is good at learning even with a small classifier layer
- Scatnet needs more complex classifier layer
- CNN reaces 99% accuracy
- Scatnet reaches 92% accuracy
- CNN is faster than Scatnet

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

# Prensentation

1. Introduction & Problem Statement (1-2 minutes)

   - Dataset overview: Lung cancer histopathological images
   - Binary classification task: adenocarcinoma vs benign
   - Project objectives

2. Data Preprocessing & Setup (2-3 minutes)

   - Dataset split methodology
   - Data augmentation decisions
   - Note about color vs grayscale (from your notes: "The model is learning the color of the class -> turning the image to gray scale")

3. Model Architectures (3-4 minutes)

   - CNN architecture details
   - ScatNet architecture
   - Classifier layer design (from your notes: "CNN is good at learning even with a small classifier layer" and "Scatnet needs more complex classifier layer")

4. Training & Evaluation (3-4 minutes)

   - K-fold cross-validation results
   - Performance metrics
   - Key findings (from your notes: "CNN reaches 99% accuracy" and "Scatnet reaches 92% accuracy")
   - Computational efficiency (from your notes: "CNN is faster than Scatnet")

5. Filter Analysis (2-3 minutes)

   - Comparison of filters from both models
   - Visual interpretation of learned features
   - Impact of data augmentation on filter quality

6. Explainable AI Results (2-3 minutes)

   - Implementation details of assigned XAI method
   - Comparison between custom implementation and Captum library
   - Attribution maps analysis

7. Conclusions (1-2 minutes)
   - Performance comparison summary
   - Key findings and insights
   - Future improvements or recommendations

Each section should include relevant visualizations (learning curves, filters, attribution maps) to support the discussion, as specifically requested in the assignment instructions.
