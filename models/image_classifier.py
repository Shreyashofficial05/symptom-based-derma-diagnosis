# image_classifier.py

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader


def train_image_model(
    train_dir,
    test_dir,
    checkpoint_path,
    epochs=5,
    batch_size=32,
    lr=1e-4
):
    # Transforms
    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Datasets & loaders
    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    test_ds = datasets.ImageFolder(test_dir, transform=test_tfms)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
    model = model.to(device)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * xb.size(0)
            running_corrects += (preds == yb).sum().item()

        epoch_loss = running_loss / len(train_ds)
        epoch_acc = running_corrects / len(train_ds)
        print(f"Epoch {epoch+1}/{epochs} — Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # Evaluation on test set
    model.eval()
    test_loss = 0.0
    test_corrects = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            preds = outputs.argmax(dim=1)
            test_loss += loss.item() * xb.size(0)
            test_corrects += (preds == yb).sum().item()

    test_loss = test_loss / len(test_ds)
    test_acc = test_corrects / len(test_ds)
    print(f"Test — Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

    # Save
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved image model to {checkpoint_path}")


if __name__ == "__main__":
    # Hardcoded paths
    base_dir = r"C:/Users/trafl/Desktop/Minor/Dataset/SkinDisease/SkinDisease"
    train_dir = os.path.join(base_dir, 'Train')
    test_dir = os.path.join(base_dir, 'Test')
    ckpt_path = r"C:/Users/trafl/Desktop/Minor/Model/image_model.pth"

    train_image_model(train_dir, test_dir, ckpt_path)