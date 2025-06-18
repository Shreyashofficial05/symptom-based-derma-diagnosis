import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from transformers import BertForSequenceClassification, BertTokenizerFast
from disease_mappings import CLASS_NAMES, name_mapping, get_display_name

class ImageClassifierWrapper(nn.Module):
    def __init__(self, ckpt_path, num_classes=len(CLASS_NAMES)):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    def predict(self, img_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(x)
            return F.softmax(outputs, dim=1).cpu().squeeze(0)

class TextClassifierWrapper:
    def __init__(self, ckpt_path, num_classes=len(CLASS_NAMES)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=num_classes
        )
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    def predict(self, text):
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(
                input_ids=enc.input_ids.to(self.device),
                attention_mask=enc.attention_mask.to(self.device)
            ).logits
            return F.softmax(logits, dim=1).cpu().squeeze(0)

def run_ensemble_model(selected_symptoms, image_path=None, w_img=0.5):
    txt_ckpt = "/Users/shreyash/Desktop/cloned copy 2/skin-disease-classification-app/models/text_model.pth"
    img_ckpt = "/Users/shreyash/Desktop/cloned copy 2/skin-disease-classification-app/models/image_model_1.pth"
    
    txt_clf = TextClassifierWrapper(txt_ckpt)
    symptoms_text = ' '.join(selected_symptoms)
    p_txt = txt_clf.predict(symptoms_text)
    
    results = []
    if image_path:
        try:
            img_clf = ImageClassifierWrapper(img_ckpt)
            p_img = img_clf.predict(image_path)
            
            top5_img = torch.topk(p_img, 5)
            top5_indices = top5_img.indices
            top5_img_scores = top5_img.values
            top5_txt_scores = p_txt[top5_indices]

            print("\nTop 5 Image Predictions:")
            for score, idx in zip(top5_img.values.tolist(), top5_indices.tolist()):
                print(f"{get_display_name(CLASS_NAMES[idx])}: {score*100:.1f}%")

            print(f"\nUsing weights: Image={w_img:.2f}, Text={1-w_img:.2f}")

            # Step 1: Raw ensemble combination
            ensemble_raw = []
            for i in range(5):
                img_score = top5_img_scores[i].item()
                txt_score = top5_txt_scores[i].item()

                if w_img == 1.0:
                    final_score = img_score
                elif w_img == 0.0:
                    final_score = txt_score
                else:
                    final_score = w_img * img_score + (1.0 - w_img) * txt_score

                ensemble_raw.append(final_score)

            # Step 2: Normalize to match image total
            raw_total = sum(ensemble_raw)
            img_total = sum(top5_img_scores).item()
            normalized_ensemble = [s * img_total / raw_total for s in ensemble_raw]

            for i, idx in enumerate(top5_indices):
                model_name = CLASS_NAMES[idx]
                display_name = get_display_name(model_name)
                img_score = top5_img_scores[i].item()
                final_score = normalized_ensemble[i]
                change = final_score - img_score

                print(f"\n{display_name}:")
                print(f"  Image score: {img_score*100:.1f}%")
                print(f"  Text score : {top5_txt_scores[i].item()*100:.1f}%")
                print(f"  Final score: {final_score*100:.1f}%")
                print(f"  Change     : {change*100:+.1f}%")

                results.append({
                    'disease': display_name,
                    'confidence': float(final_score) * 100,
                    'originalConfidence': float(img_score) * 100,
                    'change': float(change) * 100
                })

        except Exception as e:
            print(f"Error processing image: {e}")
            # Fallback to text-only
            top5 = torch.topk(p_txt, 5)
            for score, idx in zip(top5.values.tolist(), top5.indices.tolist()):
                model_name = CLASS_NAMES[idx]
                display_name = get_display_name(model_name)
                results.append({
                    'disease': display_name,
                    'confidence': float(score) * 100
                })
    else:
        print("No image provided, using text-only prediction")
        top5 = torch.topk(p_txt, 5)
        for score, idx in zip(top5.values.tolist(), top5.indices.tolist()):
            model_name = CLASS_NAMES[idx]
            display_name = get_display_name(model_name)
            results.append({
                'disease': display_name,
                'confidence': float(score) * 100
            })

    return results

def main():
    img_ckpt = "/Users/shreyash/Desktop/cloned copy 2/skin-disease-classification-app/models/image_model_1.pth"
    txt_ckpt = "/Users/shreyash/Desktop/cloned copy 2/skin-disease-classification-app/models/text_model.pth"

    img_clf = ImageClassifierWrapper(img_ckpt)
    txt_clf = TextClassifierWrapper(txt_ckpt)

    img_path = input("Enter IMAGE FILE path: ").strip()
    txt_desc = input("Enter TEXT DESCRIPTION of symptoms: ").strip()

    while True:
        try:
            w_img = float(input("Enter IMAGE weight [0.0–1.0]: ").strip())
            if 0.0 <= w_img <= 1.0:
                break
        except ValueError:
            pass
        print("Please enter a decimal between 0 and 1.")
    w_txt = 1.0 - w_img

    p_img = img_clf.predict(img_path)
    p_txt = txt_clf.predict(txt_desc)
    p_final = torch.mul(p_img, w_img) + torch.mul(p_txt, w_txt)

    idx = torch.argmax(p_final).item()
    print(f"\nEnsembled ➔ {get_display_name(CLASS_NAMES[idx])}")
    top3 = torch.topk(p_final, 3)
    for score, i in zip(top3.values.tolist(), top3.indices.tolist()):
        print(f"  {get_display_name(CLASS_NAMES[i]):25s} {score:.4f}")

if __name__ == "__main__":
    main()
