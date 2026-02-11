"""
script to test the model on images
"""
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path

from cnn_classifier import CNNClassifier, DataAugmentation, ImagePreprocessor


class ImageClassifierPredictor:
    """
    helper for making predictions
    """
    
    def __init__(self, model_path, num_classes, class_names, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names
        
        # load model
        self.model = CNNClassifier(num_classes=num_classes)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # transforms
        self.transform = DataAugmentation.get_val_transforms()
        self.preprocessor = ImagePreprocessor()
        
        print(f"model loaded from: {model_path}")
        print(f"device: {device}")
        print(f"classes: {class_names}")
    
    def predict_single_image(self, image_path, top_k=3):
        """
        predict class for a single image
        
        returns:
            predicted_class: int
            probabilities: dict of {class_name: probability}
        """
        # load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_preprocessed = self.preprocessor.preprocess_pipeline(image)
        image_pil = Image.fromarray(image_preprocessed)
        
        # transform
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        # predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # get top k predictions
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, self.num_classes))
        
        top_predictions = {}
        for prob, idx in zip(top_probs[0], top_indices[0]):
            class_name = self.class_names[idx.item()]
            top_predictions[class_name] = prob.item()
        
        predicted_class = top_indices[0][0].item()
        
        return predicted_class, top_predictions, image_preprocessed
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        visualize prediction with original image and top predictions
        """
        predicted_class, top_predictions, preprocessed_image = self.predict_single_image(image_path)
        
        # load original image
        original_image = Image.open(image_path).convert('RGB')
        
        # create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # original image
        axes[0].imshow(original_image)
        axes[0].set_title('original image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # preprocessed image
        axes[1].imshow(preprocessed_image)
        axes[1].set_title('preprocessed image', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # predictions bar chart
        classes = list(top_predictions.keys())
        probs = list(top_predictions.values())
        colors = ['green' if i == 0 else 'steelblue' for i in range(len(classes))]
        
        axes[2].barh(classes, probs, color=colors)
        axes[2].set_xlabel('probability', fontsize=12)
        axes[2].set_title('top predictions', fontsize=14, fontweight='bold')
        axes[2].set_xlim([0, 1])
        
        # add percentage labels
        for i, (class_name, prob) in enumerate(top_predictions.items()):
            axes[2].text(prob + 0.02, i, f'{prob*100:.2f}%', 
                        va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"prediction visualization saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    def visualize_feature_maps(self, image_path, layer_name='conv3', num_filters=16, save_path=None):
        """
        visualize intermediate feature maps from the cnn
        """
        # load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_preprocessed = self.preprocessor.preprocess_pipeline(image)
        image_pil = Image.fromarray(image_preprocessed)
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        # get feature maps
        with torch.no_grad():
            features = self.model.get_feature_maps(image_tensor)
        
        # select layer
        feature_map = features[layer_name][0].cpu().numpy()
        
        # visualize
        num_filters = min(num_filters, feature_map.shape[0])
        grid_size = int(np.ceil(np.sqrt(num_filters)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axes = axes.flatten()
        
        for i in range(num_filters):
            axes[i].imshow(feature_map[i], cmap='viridis')
            axes[i].set_title(f'filter {i+1}', fontsize=10)
            axes[i].axis('off')
        
        # hide unused subplots
        for i in range(num_filters, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'feature maps from {layer_name} layer', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"feature maps saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    def batch_predict(self, image_folder, output_csv='predictions.csv'):
        """
        predict on all images in a folder and save results
        """
        import csv
        
        image_paths = list(Path(image_folder).glob('*.jpg')) + \
                     list(Path(image_folder).glob('*.png')) + \
                     list(Path(image_folder).glob('*.jpeg'))
        
        results = []
        
        print(f"\nprocessing {len(image_paths)} images from {image_folder}...")
        
        for img_path in image_paths:
            predicted_class, top_predictions, _ = self.predict_single_image(str(img_path))
            
            # get top prediction
            top_class = list(top_predictions.keys())[0]
            top_prob = list(top_predictions.values())[0]
            
            results.append({
                'image_name': img_path.name,
                'predicted_class': top_class,
                'confidence': f'{top_prob*100:.2f}%',
                'top_3_predictions': str(top_predictions)
            })
        
        # save to csv
        with open(output_csv, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        
        print(f"✓ predictions saved to: {output_csv}")
        
        return results


def demo_preprocessing_pipeline(image_path):
    """
    demonstrate the preprocessing pipeline steps
    """
    preprocessor = ImagePreprocessor()
    
    # load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # apply different preprocessing steps
    filtered = preprocessor.apply_filtering(image, 'gaussian')
    contour_img, contours = preprocessor.detect_contours(image, draw=True)
    rotated = preprocessor.apply_transformations(image, rotation=15, scale=1.1)
    complete = preprocessor.preprocess_pipeline(image)
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    images = [
        (image, 'Original'),
        (filtered, 'Gaussian Filtering'),
        (contour_img, 'Contour Detection'),
        (rotated, 'Transformation (Rotate + Scale)'),
        (complete, 'Complete Preprocessing'),
    ]
    
    for idx, (img, title) in enumerate(images):
        row, col = idx // 3, idx % 3
        axes[row, col].imshow(img)
        axes[row, col].set_title(title.lower(), fontsize=14, fontweight='bold')
        axes[row, col].axis('off')
    
    # hide last subplot
    axes[1, 2].axis('off')
    
    plt.suptitle('image preprocessing pipeline', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('preprocessing_demo.png', dpi=300, bbox_inches='tight')
    print("preprocessing demo saved to: preprocessing_demo.png")
    
    return fig


if __name__ == "__main__":
    print("use this class in other scripts to predict stuff")
