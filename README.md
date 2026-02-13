# vision cnn project

a modular pytorch implementation for image classification, featuring a custom cnn architecture, automatic image preprocessing, and a complete training/evaluation pipeline. kinda designed for general use since i was just trying , it still includes a pre-configured setup for the cifar-10 dataset.

## project components

### 1. architecture & preprocessing (cnn_classifier.py)
- **model**: a deep convolutional neural network with 4 convolutional blocks, dropout layers (0.5), and fully connected layers.
- **preprocessing**: includes gaussian/bilateral filtering, canny edge detection, and contour analysis to prepare images before they hit the network.
- **augmentation**: training pipeline uses random horizontal flips, cropping, and color jittering to improve generalization.

### 2. training infrastructure (train.py)
- **trainer class**: manages the training and validation loops, learning rate scheduling (reducelronplateau), and early stopping.
- **metrics**: automatically generates classification reports, accuracy/loss curves, and confusion matrices.
- **checkpointing**: saves the `best_model.pth` based on validation accuracy and a `final_model.pth` at the end.

### 3. inference engine (inference.py)
- **predictor class**: useful for loading a trained model and making predictions on single images or batches.
- **visualizations**: can plot top-k predicted classes with confidence percentages and visualize intermediate feature maps (activations).

## project structure

- `cifar10_example.py`: main script to run the full training pipeline on cifar-10.
- `test_my_model.py`: a quick utility to load the best saved model and test it on random images from the test set.
- `results/`: directory containing all generated visualizations and performance metrics.

## getting started

### installation
```bash
pip install -r requirements.txt
```

### training
to train the model on cifar-10 with default settings:
```bash
python cifar10_example.py
```

### prediction
to predict a single image using a trained model:
```python
from inference import ImageClassifierPredictor

predictor = ImageClassifierPredictor(
    model_path='cifar10_checkpoints/best_model.pth',
    num_classes=10,
    class_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
)

# get top 3 predictions
predicted_class, top_probs, preprocessed = predictor.predict_single_image('path/to/image.jpg')
```

## training results
the `results/` folder for:
- `dataset_samples.png`: visualization of the augmented training data.
- `cifar10_training_curves.png`: evolution of loss and accuracy over epochs.
- `confusion_matrix.png`: detailed breakdown of class-wise performance.
- `test_predictions.png`: visual confirmation of the model's performance on unseen images.
