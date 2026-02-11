"""
cifar-10 training example
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from cnn_classifier import CNNClassifier, DataAugmentation
from train import Trainer, evaluate_model
from inference import ImageClassifierPredictor


def download_and_prepare_cifar10(batch_size=32):
    """
    download and prepare cifar-10 dataset
    """
    print("downloading cifar-10 dataset...")
    
    # define transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # download datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"✓ dataset loaded")
    print(f"training samples: {len(train_dataset)}")
    print(f"test samples: {len(test_dataset)}")
    print(f"classes: {class_names}")
    
    return train_loader, test_loader, class_names


def visualize_samples(data_loader, class_names, num_samples=16):
    """
    visualize sample images from the dataset
    """
    # get a batch
    images, labels = next(iter(data_loader))
    
    # denormalize images
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)
    
    # create grid
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_samples):
        img = images[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].set_title(class_names[labels[i]], fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=300, bbox_inches='tight')
    print("Sample images saved to: dataset_samples.png")
    
    return fig


def train_on_cifar10(num_epochs=50, batch_size=64, learning_rate=0.001):
    """
    complete training pipeline on cifar-10
    """
    print("\n" + "="*80)
    print("cnn training on cifar-10 dataset")
    print("="*80)
    
    # prepare data
    train_loader, test_loader, class_names = download_and_prepare_cifar10(batch_size)
    
    # split train into train/val
    train_dataset = train_loader.dataset
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    from torch.utils.data import random_split
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # visualize samples
    print("\nvisualizing dataset samples...")
    visualize_samples(train_loader, class_names)
    
    # create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNClassifier(num_classes=10, dropout=0.5)
    
    print(f"\n{'='*80}")
    print("model architecture")
    print(f"{'='*80}")
    print(model)
    print(f"\ntotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # train
    trainer = Trainer(model, device=device)
    
    print(f"\n{'='*80}")
    print("starting training")
    print(f"{'='*80}")
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        save_dir='cifar10_checkpoints',
        early_stopping_patience=15
    )
    
    # plot training curves
    print("\ngenerating training visualizations...")
    trainer.plot_training_history('cifar10_training_curves.png')
    
    # evaluate on test set
    print(f"\n{'='*80}")
    print("evaluating on test set")
    print(f"{'='*80}")
    
    # load best model
    checkpoint = torch.load('cifar10_checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_acc, confusion_mat = evaluate_model(model, test_loader, class_names, device)
    
    print(f"\n{'='*80}")
    print("training complete")
    print(f"{'='*80}")
    print(f"best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"test accuracy: {test_acc:.2f}%")
    print(f"\nmodel saved to: cifar10_checkpoints/")
    print(f"training curves: cifar10_training_curves.png")
    print(f"confusion matrix: confusion_matrix.png")
    
    return model, history, test_acc


def quick_demo(num_epochs=5):
    """
    quick demo with fewer epochs for testing
    """
    print("\n" + "="*80)
    print("quick demo 5")
    print("="*80)
    
    model, history, test_acc = train_on_cifar10(
        num_epochs=num_epochs,
        batch_size=128,
        learning_rate=0.001
    )
    
    return model, history, test_acc


if __name__ == "__main__":
    import sys
    
    print("cnn training on cifar-10")
    print("-" * 30)
    
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        # quick demo
        model, history, test_acc = quick_demo(num_epochs=5)
    else:
        # full training
        print("starting full training (50 epochs)...")
        print("this may take 30-60 minutes depending on your hardware.")
        print("\npress ctrl+c to cancel\n")
        
        try:
            model, history, test_acc = train_on_cifar10(
                num_epochs=50,
                batch_size=64,
                learning_rate=0.001
            )
        except KeyboardInterrupt:
            print("\n\ntraining interrupted by user.")
            sys.exit(0)
    
    print("\n" + "="*80)
    print("all done! check the generated files:")
    print("  - cifar10_checkpoints/best_model.pth (trained model)")
    print("  - cifar10_training_curves.png (training progress)")
    print("  - confusion_matrix.png (evaluation results)")
    print("  - dataset_samples.png (sample images)")
    print("="*80)
