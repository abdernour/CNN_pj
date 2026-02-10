import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class ImagePreprocessor:
    """
    image preprocessing with filters and transforms
    """
    
    @staticmethod
    def apply_filtering(image, filter_type='gaussian'):
        """
        apply various filters to the image
        
        args:
            image: numpy array (h, w, c)
            filter_type: 'gaussian', 'median', 'bilateral'
        """
        if filter_type == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif filter_type == 'median':
            return cv2.medianBlur(image, 5)
        elif filter_type == 'bilateral':
            return cv2.bilateralFilter(image, 9, 75, 75)
        return image
    
    @staticmethod
    def detect_contours(image, draw=False):
        """
        detect and optionally draw contours
        
        args:
            image: numpy array (h, w, c)
            draw: whether to draw contours on the image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if draw:
            output = image.copy()
            cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
            return output, contours
        
        return contours
    
    @staticmethod
    def apply_transformations(image, rotation=None, scale=None, translation=None):
        """
        apply geometric transformations
        
        args:
            image: numpy array (h, w, c)
            rotation: rotation angle in degrees
            scale: scale factor
            translation: (tx, ty) translation in pixels
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # build transformation matrix
        matrix = cv2.getRotationMatrix2D(center, rotation or 0, scale or 1.0)
        
        if translation:
            matrix[0, 2] += translation[0]
            matrix[1, 2] += translation[1]
        
        transformed = cv2.warpAffine(image, matrix, (w, h))
        return transformed
    
    @staticmethod
    def preprocess_pipeline(image, apply_filter=True, enhance_contrast=True):
        """
        complete preprocessing pipeline
        """
        # convert to numpy if pil image
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # apply filtering
        if apply_filter:
            image = ImagePreprocessor.apply_filtering(image, 'gaussian')
        
        # enhance contrast
        if enhance_contrast:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            image = cv2.merge([l, a, b])
            image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        
        return image


class DataAugmentation:
    """
    random stuff for training data
    """
    
    @staticmethod
    def get_train_transforms(img_size=224):
        """
        get training augmentation pipeline
        """
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_val_transforms(img_size=224):
        """
        get validation/test transforms (no augmentation)
        """
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class CNNClassifier(nn.Module):
    """
    basic cnn with 4 blocks and some fc layers at the end
    """
    
    def __init__(self, num_classes, input_channels=3, dropout=0.5):
        super(CNNClassifier, self).__init__()
        
        # convolutional block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2)
        )
        
        # convolutional block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.3)
        )
        
        # convolutional block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.4)
        )
        
        # convolutional block 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.4)
        )
        
        # global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # global pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # fully connected
        x = self.fc(x)
        
        return x
    
    def get_feature_maps(self, x):
        """
        extract intermediate feature maps for visualization
        """
        features = {}
        
        x1 = self.conv1(x)
        features['conv1'] = x1
        
        x2 = self.conv2(x1)
        features['conv2'] = x2
        
        x3 = self.conv3(x2)
        features['conv3'] = x3
        
        x4 = self.conv4(x3)
        features['conv4'] = x4
        
        return features


class CustomImageDataset(Dataset):
    """
    custom dataset with preprocessing pipeline
    """
    
    def __init__(self, image_paths, labels, transform=None, preprocess=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.preprocess = preprocess
        self.preprocessor = ImagePreprocessor()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # apply custom preprocessing
        if self.preprocess:
            image = self.preprocessor.preprocess_pipeline(image)
            image = Image.fromarray(image)
        
        # apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label


def visualize_augmentations(image_path, num_samples=5):
    """
    visualize the effect of data augmentation
    """
    transform = DataAugmentation.get_train_transforms()
    
    fig, axes = plt.subplots(1, num_samples + 1, figsize=(15, 3))
    
    # Original image
    original = Image.open(image_path).convert('RGB')
    axes[0].imshow(original)
    axes[0].axis('off')
    
    # augmented versions
    for i in range(num_samples):
        img = Image.open(image_path).convert('RGB')
        augmented = transform(img)
        
        # denormalize for visualization
        augmented = augmented.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        augmented = std * augmented + mean
        augmented = np.clip(augmented, 0, 1)
        
        axes[i + 1].imshow(augmented)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    return fig


def count_parameters(model):
    """
    count trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # check model
    num_classes = 10
    model = CNNClassifier(num_classes=num_classes)
    print(f"Total params: {count_parameters(model):,}")
    
    # check forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
