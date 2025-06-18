# Import from disease_mappings to ensure consistent naming
from disease_mappings import CLASS_NAMES, name_mapping, get_display_name

# Rest of the imports
import os
import ssl
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
import warnings
from tabulate import tabulate

# Disable SSL verification (only for testing purposes)
ssl._create_default_https_context = ssl._create_unverified_context

def load_model(model_path):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Initialize model architecture (same as training)
        print("Loading ResNet50 model...")
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Load the saved state dict
        print(f"Loading weights from {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        
        # Get number of classes from the state dict
        num_classes = state_dict['fc.bias'].size(0)
        print(f"Model has {num_classes} output classes")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Load the weights
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        print("Model loaded successfully")
        
        return model, device, num_classes
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def process_image(image_path):
    try:
        # Same transforms as used in training (except augmentations)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        print(f"Image loaded: {image_path} (size: {image.size})")
        image_tensor = transform(image)
        return image_tensor.unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        raise

def predict(model, image_tensor, device):
    try:
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            # Get top 5 predictions using CLASS_NAMES indices
            probs, indices = torch.topk(probabilities, k=min(5, len(CLASS_NAMES)))
            # Print terminal output with properly mapped names
            print("\nTop 5 Image Predictions:")
            for prob, idx in zip(probs[0].tolist(), indices[0].tolist()):
                class_name = CLASS_NAMES[idx]
                display_name = get_display_name(class_name)
                print(f"{display_name}: {prob*100:.1f}%")
        return indices[0].tolist(), probs[0].tolist()
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

def main():
    try:
        # Load model
        model_path = os.path.join(os.path.dirname(__file__), "image_model_1.pth")
        print(f"Looking for model at: {model_path}")
        model, device, num_classes = load_model(model_path) 
        
        # Test directory path
        test_dir = os.path.join(os.path.dirname(__file__), "../src/python_scripts/undescribed_images")
        print(f"Looking for test images in: {test_dir}")
        
        # Use CLASS_NAMES from ensemble.py for consistency
        class_names = [get_display_name(name) for name in CLASS_NAMES]

        
        # Extend class_names to match num_classes
        if len(class_names) < num_classes:
            additional_names = [f"Class_{i}" for i in range(len(class_names), num_classes)]
            class_names += additional_names
            print(f"Extended class_names list to {num_classes} classes")
        elif len(class_names) > num_classes:
            class_names = class_names[:num_classes]
            print(f"Trimmed class_names list to {num_classes} classes")
        
        # Process all images in the test directory
        if os.path.exists(test_dir):
            images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not images:
                print(f"No image files found in {test_dir}")
                return
                
            for image_file in images:
                image_path = os.path.join(test_dir, image_file)
                print(f"\n{'='*50}")
                print(f"Processing {image_file}...")
                
                # Process image
                image_tensor = process_image(image_path)
                
                # Get prediction
                pred_indices, probabilities = predict(model, image_tensor, device)
                
                # Print results in a table format
                results = []
                for idx, prob in zip(pred_indices, probabilities):
                    model_name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class_{idx}"
                    display_name = get_display_name(model_name)  # Use consistent name mapping
                    results.append([display_name, f"{prob*100:.1f}%"])  # Match frontend decimal places
                
                print("\nPrediction Results:")
                print(tabulate(results, headers=['Disease', 'Confidence'], tablefmt='grid'))
                print(f"{'='*50}\n")
        else:
            print(f"Test directory not found: {test_dir}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
