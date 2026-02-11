"""
testing the model on some images
"""
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from cnn_classifier import CNNClassifier

# class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

print("testing model")

# load the test dataset
print("\n1 loading cifar-10 test images...")
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False)

# pick a few random images to test
print("2 selecting random test images...")
test_indices = [42, 123, 567, 999, 1234, 2345, 3456, 4567, 5678]  # random picks

# load your trained model
print("3 loading your trained model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNClassifier(num_classes=10)
checkpoint = torch.load('cifar10_checkpoints/best_model.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print(f" Model loaded on {device}")

# prepare transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

print("\n4 making predictions...")
print("="*70)

# create a figure to show results
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

correct_predictions = 0
total_predictions = 0

for idx, test_idx in enumerate(test_indices):
    # get image and true label
    image, true_label = test_dataset[test_idx]
    
    # prepare for model
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # predict
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_label = predicted.item()
    confidence_percent = confidence.item() * 100
    true_label_name = class_names[true_label]
    predicted_label_name = class_names[predicted_label]
    
    # track accuracy
    total_predictions += 1
    if predicted_label == true_label:
        correct_predictions += 1
        result = "correct"
        color = 'green'
    else:
        result = "wrong"
        color = 'red'
    
    # print result
    print(f"\nimage #{test_idx}:")
    print(f"  true label:      {true_label_name}")
    print(f"  ai predicted:    {predicted_label_name}")
    print(f"  confidence:      {confidence_percent:.1f}%")
    print(f"  result:          {result}")
    
    # show image
    axes[idx].imshow(image)
    axes[idx].set_title(
        f"true: {true_label_name}\n"
        f"predicted: {predicted_label_name}\n"
        f"confidence: {confidence_percent:.1f}%",
        fontsize=10,
        color=color,
        fontweight='bold'
    )
    axes[idx].axis('off')

# calculate accuracy
accuracy = (correct_predictions / total_predictions) * 100

print("\n" + "="*70)
print(f"results: {correct_predictions}/{total_predictions} correct predictions ({accuracy:.1f}% accuracy)")
print("="*70)

# save the visualization
plt.tight_layout()
plt.savefig('test_predictions.png', dpi=300, bbox_inches='tight')
print("\npredictions saved to: test_predictions.png")
print("\nopen 'test_predictions.png' to see all predictions with images")
